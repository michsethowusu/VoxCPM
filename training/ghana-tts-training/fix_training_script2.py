#!/usr/bin/env python3
"""Modify the tokenize + val_texts section to handle cached dataset."""
import sys

SCRIPT = "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/scripts/train_voxcpm_finetune.py"

with open(SCRIPT, "r") as f:
    content = f.read()

# Fix1: modify ghana-latents branch (already done by previous script, skip if already modified)
old1 = '''    if train_manifest == "ghana-latents":
        # Precomputed VoxCPM AudioVAE latents (parquet): feat + tagged text + dataset_id + split.
        import glob as _glob
        from datasets import load_dataset as _ld
        _ldir = os.environ.get("GHANA_LATENTS_DIR", "/mnt/volume_d2wey28/data/voxcpm-latents")
        _files = sorted(_glob.glob(_ldir + "/*.parquet"))
        _ds = _ld("parquet", data_files=_files, split="train")
        train_ds = _ds.filter(lambda r: r["split"] == "train")
        val_ds = _ds.filter(lambda r: r["split"] == "dev")
        print(f"[ghana-latents] train={len(train_ds)} val={len(val_ds)}", file=sys.stderr)'''

new1 = '''    if train_manifest == "ghana-latents":
        # Precomputed VoxCPM AudioVAE latents (parquet): feat + tagged text + dataset_id + split.
        # Check for pre-cached (tokenized) dataset first.
        import os as _os
        _cached = "/mnt/volume_d2wey28/data/voxcpm-latents-cached"
        _train_cache = _os.path.join(_cached, "train")
        _dev_cache = _os.path.join(_cached, "dev")
        if _os.path.isdir(_train_cache) and _os.path.isdir(_dev_cache):
            from datasets import load_from_disk as _lfd
            train_ds = _lfd(_train_cache)
            val_ds = _lfd(_dev_cache)
            print(f"[ghana-latents] loaded cached: train={len(train_ds)} val={len(val_ds)}", file=sys.stderr)
        else:
            import glob as _glob
            from datasets import load_dataset as _ld
            _ldir = _os.environ.get("GHANA_LATENTS_DIR", "/mnt/volume_d2wey28/data/voxcpm-latents")
            _files = sorted(_glob.glob(_ldir + "/*.parquet"))
            _ds = _ld("parquet", data_files=_files, split="train")
            train_ds = _ds.filter(lambda r: r["split"] == "train")
            val_ds = _ds.filter(lambda r: r["split"] == "dev")
            print(f"[ghana-latents] train={len(train_ds)} val={len(val_ds)}", file=sys.stderr)'''

if old1 in content:
    content = content.replace(old1, new1, 1)
    print("Fix1: updated ghana-latents branch (added cached check)")
else:
    print("Fix1: ghana-latents branch already updated or not found, skipping")

# Fix2: tokenize step - skip if text_ids already exist
old2 = '''    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    # Save original validation texts for audio generation display
    val_texts = None
    if val_ds is not None:
        val_texts = list(val_ds["text"])  # Save original texts'''

new2 = '''    # Tokenize text (skip if cached dataset already has text_ids)
    if "text_ids" not in train_ds.column_names:
        def tokenize(batch):
            text_list = batch["text"]
            text_ids = [tokenizer(text) for text in text_list]
            return {"text_ids": text_ids}

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        # Save original validation texts for audio generation display
        val_texts = None
        if val_ds is not None:
            val_texts = list(val_ds["text"])  # Save original texts
    else:
        # Already tokenized (cached); reconstruct val_texts from text_ids for audio gen
        val_texts = None
        if val_ds is not None:
            _raw = val_ds["text_ids"]
            val_texts = [tokenizer.decode(t, skip_special_tokens=False) for t in _raw]'''

assert old2 in content, "Could not find the tokenize section"
content = content.replace(old2, new2, 1)
print("Fix2: updated tokenize step")

with open(SCRIPT, "w") as f:
    f.write(content)
print("OK: training script updated successfully")
