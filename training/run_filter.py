#!/usr/bin/env python3
"""
Full quality-filter pass over ghana-english-asr-2700hrs.

CPU-only, file-parallel (leaves the GPU free for latent precompute):
each worker takes whole parquet shards, decodes audio, scores with
DNSMOS (SIG/BAK/OVRL) + PANNs CNN14 (music/applause/crowd), applies the
locked "very clean" thresholds, and writes a filtered shard keeping the
original audio + text + duration + all score columns. Resumable: a shard
whose output already exists is skipped.

Locked thresholds (OVRL x BAK dominant; music/applause belt-and-suspenders):
  ovrl>=3.1, bak>=3.8, sig>=3.0, music<=0.2, applause<=0.2, speech>=0.5,
  duration in [2,20]s, n_chars>=8, char_rate in [3,30]
"""
import os
# Force single-threaded math per worker BEFORE importing torch/onnxruntime/numpy,
# so N worker processes use ~N cores instead of N*cores threads (load explosion).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import io, sys, glob, json, argparse, time, traceback
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

DNSMOS_SR = 16000
PANNS_SR = 32000
TH = dict(ovrl=3.1, bak=3.8, sig=3.0, music=0.2, applause=0.2, speech=0.5,
          dur_min=2.0, dur_max=20.0, min_chars=8, char_rate_min=3.0, char_rate_max=30.0)

ARTIFACT_CLASSES = ["Music", "Musical instrument", "Singing",
                    "Applause", "Clapping", "Cheering", "Crowd"]

# ---- per-worker singletons -------------------------------------------------
_dns = None
_panns = None
_idx = None

def _init_worker():
    global _dns, _panns, _idx
    import torch
    torch.set_num_threads(1)  # PANNs CPU inference: one thread per worker
    from speechmos import dnsmos
    _dns = dnsmos
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    _panns = AudioTagging(checkpoint_path=None, device="cpu")
    _idx = {c: labels.index(c) for c in ARTIFACT_CLASSES if c in labels}
    _idx["_Speech"] = labels.index("Speech") if "Speech" in labels else -1

def _resample(a, sr, tgt):
    if sr == tgt:
        return a
    import librosa
    return librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=tgt)

def _score_batch(audios16, srs):
    """audios16: list of 16k float32 arrays. Returns list of score dicts."""
    out = []
    # PANNs: pad to max len in batch at 32k, single inference call
    a32 = [_resample(a, sr, PANNS_SR) for a, sr in zip(audios16, srs)]
    maxlen = max(len(x) for x in a32)
    batch = np.zeros((len(a32), maxlen), dtype=np.float32)
    for i, x in enumerate(a32):
        batch[i, :len(x)] = x
    clip, _ = _panns.inference(batch)  # (B, 527)
    for i, a in enumerate(audios16):
        d = {}
        try:
            d = _dns.run(a, sr=DNSMOS_SR)
        except Exception:
            pass
        sig = float(d.get("sig_mos", d.get("SIG", np.nan)))
        bak = float(d.get("bak_mos", d.get("BAK", np.nan)))
        ovrl = float(d.get("ovrl_mos", d.get("OVRL", np.nan)))
        p808 = float(d.get("p808_mos", d.get("P808_MOS", np.nan)))
        c = clip[i]
        music = float(max(c[_idx[k]] for k in ("Music", "Singing", "Musical instrument") if k in _idx))
        applause = float(max(c[_idx[k]] for k in ("Applause", "Clapping", "Cheering", "Crowd") if k in _idx))
        speech = float(c[_idx["_Speech"]]) if _idx["_Speech"] >= 0 else float("nan")
        out.append(dict(sig=sig, bak=bak, ovrl=ovrl, p808=p808,
                        music_prob=music, applause_prob=applause, speech_prob=speech))
    return out

def _passes(s, dur, txt):
    n = len(txt.strip())
    cr = n / dur if dur > 0 else 0
    return (s["ovrl"] >= TH["ovrl"] and s["bak"] >= TH["bak"] and s["sig"] >= TH["sig"]
            and s["music_prob"] <= TH["music"] and s["applause_prob"] <= TH["applause"]
            and s["speech_prob"] >= TH["speech"]
            and TH["dur_min"] <= dur <= TH["dur_max"]
            and n >= TH["min_chars"] and TH["char_rate_min"] <= cr <= TH["char_rate_max"])

def process_file(in_path, out_path, batch=24):
    if os.path.exists(out_path):
        return (in_path, "skip", 0, 0)
    tmp = out_path + ".tmp"
    pf = pq.ParquetFile(in_path)
    kept_audio, kept_text, kept_dur = [], [], []
    kept_scores = {k: [] for k in ["sig", "bak", "ovrl", "p808", "music_prob", "applause_prob", "speech_prob"]}
    total = 0
    buf_audio, buf_bytes, buf_srs, buf_text, buf_dur = [], [], [], [], []

    def flush():
        if not buf_audio:
            return
        scores = _score_batch(buf_audio, buf_srs)
        for s, ab, txt, dur in zip(scores, buf_bytes, buf_text, buf_dur):
            if _passes(s, dur, txt):
                kept_audio.append({"bytes": ab, "path": None})
                kept_text.append(txt)
                kept_dur.append(dur)
                for k in kept_scores:
                    kept_scores[k].append(round(s[k], 4))
        buf_audio.clear(); buf_bytes.clear(); buf_srs.clear(); buf_text.clear(); buf_dur.clear()

    for rb in pf.iter_batches(batch_size=512):
        d = rb.to_pylist()
        for r in d:
            total += 1
            ab = r["audio"]["bytes"]
            try:
                a, sr = sf.read(io.BytesIO(ab), dtype="float32")
            except Exception:
                continue
            if a.ndim > 1:
                a = a.mean(axis=1)
            buf_audio.append(a); buf_bytes.append(ab); buf_srs.append(sr)
            buf_text.append(r.get("corrected_text", "") or "")
            buf_dur.append(float(r.get("duration_ss", len(a) / sr)))
            if len(buf_audio) >= batch:
                flush()
        flush()

    tbl = pa.table({
        "audio": pa.array(kept_audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "corrected_text": pa.array(kept_text, type=pa.string()),
        "duration_ss": pa.array(kept_dur, type=pa.float32()),
        **{k: pa.array(v, type=pa.float32()) for k, v in kept_scores.items()},
    })
    pq.write_table(tbl, tmp)
    os.replace(tmp, out_path)
    return (in_path, "ok", total, len(kept_text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/mnt/volume_d2wey28/data/ghana-english-asr-2700hrs/data")
    ap.add_argument("--out-dir", default="/mnt/volume_d2wey28/data/ghana-english-tts-filtered/data")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit-files", type=int, default=0, help="process only N files (benchmark)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.parquet")))
    if args.limit_files:
        files = files[:args.limit_files]
    jobs = [(f, os.path.join(args.out_dir, "filtered-" + os.path.basename(f))) for f in files]
    print(f"{len(jobs)} shards, {args.workers} workers", flush=True)
    t0 = time.time(); tot_in = tot_keep = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(process_file, i, o): i for i, o in jobs}
        for k, fut in enumerate(as_completed(futs)):
            try:
                ip, st, n, kept = fut.result()
            except Exception as e:
                print("ERROR", futs[fut], e, traceback.format_exc(), flush=True); continue
            tot_in += n; tot_keep += kept
            el = time.time() - t0
            print(f"[{k+1}/{len(jobs)}] {os.path.basename(ip)} {st} in={n} kept={kept} "
                  f"({100*kept/n if n else 0:.1f}%) | cum {tot_keep}/{tot_in} ({100*tot_keep/tot_in if tot_in else 0:.1f}%) | {el/60:.1f}min", flush=True)
    print(f"DONE kept {tot_keep}/{tot_in} = {100*tot_keep/tot_in if tot_in else 0:.1f}% in {(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
