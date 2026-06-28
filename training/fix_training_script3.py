#!/usr/bin/env python3
"""Fix the tokenize/val_texts/val_ds section structure."""
import sys

SCRIPT = "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/scripts/train_voxcpm_finetune.py"

with open(SCRIPT, "r") as f:
    content = f.read()

old = '''    # Tokenize text (skip if cached dataset already has text_ids)
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
            val_texts = [tokenizer.decode(t, skip_special_tokens=False) for t in _raw]
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])'''

new = '''    # Tokenize text (skip if cached dataset already has text_ids)
    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    if "text_ids" not in train_ds.column_names:
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
            val_texts = [tokenizer.decode(list(t), skip_special_tokens=False) for t in _raw]

    # Tokenize val_ds (if tokenization not already done)
    if val_ds is not None and "text_ids" not in val_ds.column_names:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])'''

assert old in content, "Could not find the block to replace"
content = content.replace(old, new, 1)
with open(SCRIPT, "w") as f:
    f.write(content)
print("OK: fixed tokenize/val_ds section")
