#!/usr/bin/env python3
"""
Fast TTS-quality filter for ghana-english-asr-2700hrs.

Per the chosen approach: WebRTC VAD (speech presence) + loudness (dBFS),
all CPU, ~600 clips/s/core. Keeps clips that are predominantly human speech
at a sane level; rejects silence, near-silent, noise/music-dominated, clipped.

Keeps audio + text + duration and records speech_ratio + dbfs as columns.
Resumable: a shard whose output exists is skipped.
"""
import os, io, sys, glob, argparse, time, traceback
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
import webrtcvad
from concurrent.futures import ProcessPoolExecutor, as_completed

# Locked thresholds (from calibration distributions on 300+ clips)
TH = dict(speech_ratio=0.75, dbfs_min=-30.0, dbfs_max=-12.0,
          dur_min=2.0, dur_max=20.0, min_chars=8)
VAD_AGGR = 2          # 0..3 (2 = balanced)
FRAME_MS = 30

_vad = None
def _init():
    global _vad
    _vad = webrtcvad.Vad(VAD_AGGR)

def _dbfs(a):
    return 20.0 * np.log10(np.sqrt(np.mean(a**2) + 1e-12) + 1e-12)

def _speech_ratio(a, sr=16000):
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()
    fl = int(sr * FRAME_MS / 1000) * 2
    n = tot = 0
    for i in range(0, len(pcm) - fl, fl):
        tot += 1
        if _vad.is_speech(pcm[i:i+fl], sr):
            n += 1
    return (n / tot) if tot else 0.0

def process_file(in_path, out_path):
    if os.path.exists(out_path):
        return (in_path, "skip", 0, 0)
    if _vad is None:
        _init()
    pf = pq.ParquetFile(in_path)
    keep_audio, keep_text, keep_dur, keep_sr, keep_db = [], [], [], [], []
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
            srat = _speech_ratio(a, sr)
            if srat < TH["speech_ratio"]:
                continue
            keep_audio.append({"bytes": ab, "path": None})
            keep_text.append(txt); keep_dur.append(float(r.get("duration_ss", dur)))
            keep_sr.append(round(srat, 4)); keep_db.append(round(float(db), 2))
    tmp = out_path + ".tmp"
    tbl = pa.table({
        "audio": pa.array(keep_audio, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "corrected_text": pa.array(keep_text, type=pa.string()),
        "duration_ss": pa.array(keep_dur, type=pa.float32()),
        "speech_ratio": pa.array(keep_sr, type=pa.float32()),
        "dbfs": pa.array(keep_db, type=pa.float32()),
    })
    pq.write_table(tbl, tmp)
    os.replace(tmp, out_path)
    return (in_path, "ok", total, len(keep_text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/mnt/volume_d2wey28/data/ghana-english-asr-2700hrs/data")
    ap.add_argument("--out-dir", default="/mnt/volume_d2wey28/data/ghana-english-tts-filtered/data")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.parquet")))
    jobs = [(f, os.path.join(args.out_dir, "filtered-" + os.path.basename(f))) for f in files]
    print(f"{len(jobs)} shards, {args.workers} workers", flush=True)
    t0 = time.time(); tin = tk = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        futs = {ex.submit(process_file, i, o): i for i, o in jobs}
        for k, fut in enumerate(as_completed(futs)):
            ip, st, n, kept = fut.result()
            tin += n; tk += kept
            print(f"[{k+1}/{len(jobs)}] {os.path.basename(ip)} {st} {kept}/{n} "
                  f"({100*kept/n if n else 0:.0f}%) | cum {tk}/{tin} ({100*tk/tin if tin else 0:.1f}%) | {time.time()-t0:.0f}s", flush=True)
    print(f"DONE kept {tk}/{tin} = {100*tk/tin if tin else 0:.1f}% in {(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
