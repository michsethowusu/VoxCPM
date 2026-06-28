#!/usr/bin/env python3
"""
Quality-filter the Ghana English ASR corpus into a TTS-grade subset.

Two modes:
  calibrate  -- score a random sample, dump score distributions + example clips
               at different quality bands, so thresholds can be chosen by ear/eye.
  run        -- score the full corpus, keep segments passing the locked thresholds,
               write a parquet manifest (scores kept as columns) ready to push to HF.

Scorers (audio-quality only, per locked decision — no transcript/Whisper pass):
  - DNSMOS P.835 (SIG / BAK / OVRL) via `speechmos`  -> clarity, background, overall
  - PANNs CNN14 AudioSet tagging                     -> music/applause/clapping/crowd probs
  - structural                                       -> duration window, text sanity
"""
import os, sys, csv, json, argparse, random
import numpy as np

SRC_DATASET = "ghananlpcommunity/ghana-english-asr-2700hrs"
DNSMOS_SR = 16000
PANNS_SR = 32000

# AudioSet classes we treat as disqualifying "non-clean-speech" artifacts
ARTIFACT_CLASSES = [
    "Music", "Musical instrument", "Singing", "Applause", "Clapping",
    "Cheering", "Crowd", "Hubbub, speech noise, speech babble", "Inside, small room",
]

# ---- lazy globals (loaded once) ------------------------------------------
_dnsmos = None
_panns = None
_panns_labels = None
_artifact_idx = None


def load_scorers(device="cuda"):
    global _dnsmos, _panns, _panns_labels, _artifact_idx
    if _dnsmos is None:
        from speechmos import dnsmos
        _dnsmos = dnsmos
    if _panns is None:
        from panns_inference import AudioTagging
        from panns_inference.config import labels
        _panns = AudioTagging(checkpoint_path=None, device=device)
        _panns_labels = labels
        idx = {}
        for cls in ARTIFACT_CLASSES:
            if cls in labels:
                idx[cls] = labels.index(cls)
        _artifact_idx = idx
        # also locate Speech for a speech-presence sanity signal
        _artifact_idx["_Speech"] = labels.index("Speech") if "Speech" in labels else -1


def resample(audio, sr, target):
    if sr == target:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target)


def score_clip(audio, sr, text):
    """Return a dict of all quality signals for one clip."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr

    # DNSMOS (needs 16k)
    a16 = resample(audio, sr, DNSMOS_SR)
    try:
        d = _dnsmos.run(a16, sr=DNSMOS_SR)
    except Exception as e:
        d = {}
    sig = float(d.get("sig_mos", d.get("SIG", d.get("SIG_MOS", np.nan))))
    bak = float(d.get("bak_mos", d.get("BAK", d.get("BAK_MOS", np.nan))))
    ovrl = float(d.get("ovrl_mos", d.get("OVRL", d.get("OVRL_MOS", np.nan))))
    p808 = float(d.get("p808_mos", d.get("P808_MOS", np.nan)))

    # PANNs tagging (needs 32k)
    a32 = resample(audio, sr, PANNS_SR)
    clip, _ = _panns.inference(a32[None, :])
    clip = clip[0]
    arts = {cls: float(clip[i]) for cls, i in _artifact_idx.items() if cls != "_Speech" and i >= 0}
    speech_prob = float(clip[_artifact_idx["_Speech"]]) if _artifact_idx["_Speech"] >= 0 else np.nan
    artifact_max = max(arts.values()) if arts else 0.0
    music_prob = max(arts.get("Music", 0.0), arts.get("Singing", 0.0), arts.get("Musical instrument", 0.0))
    applause_prob = max(arts.get("Applause", 0.0), arts.get("Clapping", 0.0),
                        arts.get("Cheering", 0.0), arts.get("Crowd", 0.0))

    # structural
    txt = (text or "").strip()
    n_chars = len(txt)
    char_rate = n_chars / dur if dur > 0 else 0.0

    return {
        "duration": round(dur, 3),
        "sig": round(sig, 3), "bak": round(bak, 3), "ovrl": round(ovrl, 3), "p808": round(p808, 3),
        "music_prob": round(music_prob, 4), "applause_prob": round(applause_prob, 4),
        "artifact_max": round(artifact_max, 4), "speech_prob": round(speech_prob, 4),
        "n_chars": n_chars, "char_rate": round(char_rate, 2),
    }


def passes(s, th):
    return (
        s["ovrl"] >= th["ovrl"] and
        s["sig"] >= th["sig"] and
        s["bak"] >= th["bak"] and
        s["music_prob"] <= th["music_prob"] and
        s["applause_prob"] <= th["applause_prob"] and
        s["speech_prob"] >= th["speech_prob"] and
        th["dur_min"] <= s["duration"] <= th["dur_max"] and
        s["n_chars"] >= th["min_chars"] and
        th["char_rate_min"] <= s["char_rate"] <= th["char_rate_max"]
    )


def pctiles(vals, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    vals = np.array([v for v in vals if not np.isnan(v)], dtype=np.float64)
    if len(vals) == 0:
        return {p: float("nan") for p in ps}
    return {p: round(float(np.percentile(vals, p)), 3) for p in ps}


def calibrate(n, out_dir, buffer_size, seed, dump_examples):
    import soundfile as sf
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    load_scorers()
    print(f"Streaming {n} random samples from {SRC_DATASET} (shuffle buffer {buffer_size})...", flush=True)
    ds = load_dataset(SRC_DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    rows = []
    ex_dir = os.path.join(out_dir, "examples")
    if dump_examples:
        os.makedirs(ex_dir, exist_ok=True)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        text = ex.get("corrected_text", "")
        s = score_clip(audio, sr, text)
        s["idx"] = i
        s["text"] = (text or "")[:120]
        rows.append(s)
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{n}", flush=True)
        # dump a handful of example clips bucketed by OVRL for listening
        if dump_examples and i < 4000:
            band = "good" if s["ovrl"] >= 3.0 else ("mid" if s["ovrl"] >= 2.5 else "bad")
            bdir = os.path.join(ex_dir, band)
            os.makedirs(bdir, exist_ok=True)
            existing = len(os.listdir(bdir))
            if existing < 15:
                fn = f"{band}_{i:05d}_ovrl{s['ovrl']:.2f}_bak{s['bak']:.2f}_mus{s['music_prob']:.2f}.wav"
                try:
                    sf.write(os.path.join(bdir, fn), resample(np.asarray(audio, np.float32), sr, 16000), 16000)
                except Exception:
                    pass

    # write raw scores
    csv_path = os.path.join(out_dir, "calibration_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # distributions
    print("\n===== SCORE DISTRIBUTIONS (n={}) =====".format(len(rows)))
    for key in ["ovrl", "sig", "bak", "p808", "music_prob", "applause_prob", "speech_prob", "duration", "char_rate"]:
        print(f"  {key:14}", pctiles([r[key] for r in rows]))

    # yield rates under a few candidate threshold sets
    candidates = {
        "lenient": dict(ovrl=2.8, sig=3.0, bak=3.0, music_prob=0.5, applause_prob=0.5, speech_prob=0.2,
                        dur_min=2.0, dur_max=20.0, min_chars=5, char_rate_min=3.0, char_rate_max=35.0),
        "balanced": dict(ovrl=3.0, sig=3.2, bak=3.5, music_prob=0.3, applause_prob=0.3, speech_prob=0.4,
                         dur_min=2.0, dur_max=18.0, min_chars=8, char_rate_min=4.0, char_rate_max=30.0),
        "strict": dict(ovrl=3.2, sig=3.5, bak=4.0, music_prob=0.15, applause_prob=0.15, speech_prob=0.6,
                       dur_min=2.5, dur_max=15.0, min_chars=10, char_rate_min=5.0, char_rate_max=28.0),
    }
    print("\n===== YIELD UNDER CANDIDATE THRESHOLDS =====")
    for name, th in candidates.items():
        kept = sum(1 for r in rows if passes(r, th))
        print(f"  {name:9} keep {kept:5d}/{len(rows)} = {100*kept/len(rows):5.1f}%")
    with open(os.path.join(out_dir, "candidate_thresholds.json"), "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nScores -> {csv_path}")
    if dump_examples:
        print(f"Example clips -> {ex_dir}/(good|mid|bad)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "run"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--buffer-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/mnt/volume_d2wey28/projects/voxcpm-ghana/english_filter")
    ap.add_argument("--no-examples", action="store_true")
    args = ap.parse_args()

    if args.mode == "calibrate":
        calibrate(args.n, args.out_dir, args.buffer_size, args.seed, dump_examples=not args.no_examples)
    else:
        print("run mode is wired after thresholds are locked in calibration", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
