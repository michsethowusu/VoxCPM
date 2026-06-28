"""Precompute VoxCPM AudioVAE latents -> parquet shards (feat fp16 + tagged text
+ dataset_id + balanced split). ~40GB total, no 760GB on-the-fly cache."""
import os, io, glob, time
import numpy as np, soundfile as sf, torch
import pyarrow as pa, pyarrow.parquet as pq
from voxcpm.model.voxcpm import VoxCPMModel

BASE = "/mnt/volume_d2wey28/models/VoxCPM-0.5B"
GHANA = "/mnt/volume_d2wey28/data/ghana-speech"
ENGLISH = "/mnt/volume_d2wey28/data/ghana-english-tts-filtered/data"
OUT = "/mnt/volume_d2wey28/data/voxcpm-latents"
DEV_PER_LANG, BATCH, D = 40, 32, 64

def code_for_dir(name):
    if name.startswith("Akuapem_Twi"): return "twi-akuapem"
    if name.startswith("Asante_Twi"):  return "twi-asante"
    return name.split("_")[-1]

def main():
    os.makedirs(OUT, exist_ok=True)
    m = VoxCPMModel.from_local(BASE, optimize=False, training=False)
    vae = m.audio_vae.to("cuda").eval()
    SR, hop = vae.sample_rate, vae.hop_length

    cfgs = sorted(glob.glob(GHANA + "/*/"))
    codes = sorted({code_for_dir(os.path.basename(d.rstrip("/"))) for d in cfgs} | {"en"})
    did = {c: i for i, c in enumerate(codes)}
    sources = [(d, code_for_dir(os.path.basename(d.rstrip("/"))), "text") for d in cfgs]
    sources.append((ENGLISH + "/", "en", "corrected_text"))

    def encode_batch(wavs):
        maxlen = max(w.shape[0] for w in wavs)
        if maxlen % hop: maxlen += hop - maxlen % hop
        b = torch.zeros(len(wavs), 1, maxlen, dtype=torch.float32, device="cuda")
        tl = []
        for i, w in enumerate(wavs):
            b[i, 0, :w.shape[0]] = torch.from_numpy(w); tl.append(int(np.ceil(w.shape[0] / hop)))
        with torch.no_grad():
            z = vae.encode(b, SR).transpose(1, 2)
        return [z[i, :tl[i]].half().cpu().numpy() for i in range(len(wavs))]

    t0 = time.time(); total = 0
    for src, code, tcol in sources:
        files = sorted(glob.glob(src + "*.parquet"))
        tag = f"<|lang:{code}|> "
        dev_taken = 0  # per-code dev counter across this code's files
        for f in files:
            out_f = os.path.join(OUT, f"{code}__{os.path.basename(f)}")
            if os.path.exists(out_f):
                continue
            feats, ts, texts, dids, splits = [], [], [], [], []
            buf_w, buf_t = [], []
            def flush():
                nonlocal dev_taken
                if not buf_w: return
                for feat, txt in zip(encode_batch(buf_w), buf_t):
                    feats.append(feat.tobytes()); ts.append(feat.shape[0])
                    texts.append(txt); dids.append(did[code])
                    if dev_taken < DEV_PER_LANG:
                        splits.append("dev"); dev_taken += 1
                    else:
                        splits.append("train")
                buf_w.clear(); buf_t.clear()
            for rb in pq.ParquetFile(f).iter_batches(batch_size=512):
                for r in rb.to_pylist():
                    ab = r["audio"].get("bytes") if isinstance(r["audio"], dict) else None
                    txt = (r.get(tcol) or "").strip()
                    if not ab or not txt: continue
                    try:
                        a, sr = sf.read(io.BytesIO(ab), dtype="float32")
                    except Exception: continue
                    if a.ndim > 1: a = a.mean(axis=1)
                    buf_w.append(a); buf_t.append(tag + txt)
                    if len(buf_w) >= BATCH: flush()
                flush()
            pq.write_table(pa.table({
                "feat": pa.array(feats, type=pa.binary()),
                "feat_t": pa.array(ts, type=pa.int32()),
                "text": pa.array(texts, type=pa.string()),
                "dataset_id": pa.array(dids, type=pa.int32()),
                "split": pa.array(splits, type=pa.string()),
            }), out_f)
            total += len(feats)
            print(f"[{code}] {os.path.basename(f)} -> {len(feats)} (dev_taken={dev_taken}) | total {total} | {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"DONE {total} clips, {(time.time()-t0)/60:.1f}min, dataset_ids={did}", flush=True)

if __name__ == "__main__":
    main()
