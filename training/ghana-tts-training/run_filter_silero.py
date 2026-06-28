#!/usr/bin/env python3
"""
Higher-quality TTS filter for ghana-english-asr-2700hrs using Silero VAD.

Per clip: Silero per-frame speech probability over 512-sample windows ->
mean_speech_prob; plus RMS dBFS loudness. Keep clips with confident speech
(mean_prob >= 0.85) at a sane level. ~12 clips/s/core, 16 workers ~= 1h.

Keeps audio+text+duration, records mean_speech_prob + dbfs as columns.
Resumable: a shard whose output exists is skipped.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import io, glob, argparse, time
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

TH = dict(mean_prob=0.85, dbfs_min=-30.0, dbfs_max=-12.0,
          dur_min=2.0, dur_max=20.0, min_chars=8)

_model = None
def _init():
    global _model
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _model = load_silero_vad()

def _dbfs(a):
    return 20.0 * np.log10(np.sqrt(np.mean(a**2) + 1e-12) + 1e-12)

def _mean_speech_prob(a):
    import torch
    t = torch.from_numpy(a)
    _model.reset_states()
    probs = []
    for i in range(0, len(t) - 512, 512):
        probs.append(_model(t[i:i+512], 16000).item())
    return float(np.mean(probs)) if probs else 0.0

def process_file(in_path, out_path):
    if os.path.exists(out_path):
        return (in_path, "skip", 0, 0)
    if _model is None:
        _init()
    pf = pq.ParquetFile(in_path)
    ka, kt, kd, kp, kdb = [], [], [], [], []
    total = 0
    for rb in pf.iter_batches(batch_size=512):
        for r in rb.to_pylist():
            total += 1
            ab = r["audio"].get("bytes") if isinstance(r["audio"], dict) else None
            txt = (r.get("corrected_text") or "").strip()
            if not ab or len(txt) < TH["min_chars"]:
                continue
            try:
                a, sr = sf.read(io.BytesIO(ab), dtype="float32")
            except Exception:
                continue
            if a.ndim > 1:
                a = a.mean(axis=1)
            dur = len(a) / sr
            if not (TH["dur_min"] <= dur <= TH["dur_max"]):
                continue
            db = _dbfs(a)
            if not (TH["dbfs_min"] <= db <= TH["dbfs_max"]):
                continue
            mp = _mean_speech_prob(a)
            if mp < TH["mean_prob"]:
                continue
            ka.append({"bytes": ab, "path": None})
            kt.append(txt); kd.append(float(r.get("duration_ss", dur)))
            kp.append(round(mp, 4)); kdb.append(round(float(db), 2))
    tmp = out_path + ".tmp"
    pq.write_table(pa.table({
        "audio": pa.array(ka, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "corrected_text": pa.array(kt, type=pa.string()),
        "duration_ss": pa.array(kd, type=pa.float32()),
        "mean_speech_prob": pa.array(kp, type=pa.float32()),
        "dbfs": pa.array(kdb, type=pa.float32()),
    }), tmp)
    os.replace(tmp, out_path)
    return (in_path, "ok", total, len(kt))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/mnt/volume_d2wey28/data/ghana-english-asr-2700hrs/data")
    ap.add_argument("--out-dir", default="/mnt/volume_d2wey28/data/ghana-english-tts-filtered/data")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.parquet")))
    jobs = [(f, os.path.join(args.out_dir, "filtered-" + os.path.basename(f))) for f in files]
    print(f"{len(jobs)} shards, {args.workers} workers, mean_prob>={TH['mean_prob']}", flush=True)
    t0 = time.time(); tin = tk = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        futs = {ex.submit(process_file, i, o): i for i, o in jobs}
        for k, fut in enumerate(as_completed(futs)):
            ip, st, n, kept = fut.result()
            tin += n; tk += kept
            print(f"[{k+1}/{len(jobs)}] {os.path.basename(ip)} {st} {kept}/{n} "
                  f"({100*kept/n if n else 0:.0f}%) | cum {tk}/{tin} ({100*tk/tin if tin else 0:.1f}%) | {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"DONE kept {tk}/{tin} = {100*tk/tin if tin else 0:.1f}% in {(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
