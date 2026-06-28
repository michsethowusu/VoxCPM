#!/usr/bin/env python3
"""
Pre-process VoxCPM latents: tokenize text, cache to Arrow on disk.

This eliminates the 2200-parquet-file I/O overhead during training.
Run once before training.

Output: /mnt/volume_d2wey28/data/voxcpm-latents-cached/{train,dev}/
"""
import os, glob, sys, time
sys.path.insert(0, "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/src")

import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from voxcpm.model.voxcpm import VoxCPMModel

BASE = "/mnt/volume_d2wey28/models/VoxCPM-0.5B"
LDIR = os.environ.get("GHANA_LATENTS_DIR", "/mnt/volume_d2wey28/data/voxcpm-latents")
CACHED = "/mnt/volume_d2wey28/data/voxcpm-latents-cached"

def main():
    t0 = time.time()

    # --- Load all parquet files ---
    files = sorted(glob.glob(LDIR + "/*.parquet"))
    print(f"Loading {len(files)} parquet files ...", flush=True)
    ds = load_dataset("parquet", data_files=files, split="train")
    print(f"Loaded {len(ds)} rows in {time.time()-t0:.1f}s", flush=True)

    # --- Filter by split ---
    train_ds = ds.filter(lambda r: r["split"] == "train")
    val_ds = ds.filter(lambda r: r["split"] == "dev")
    print(f"Train: {len(train_ds)}, Dev: {len(val_ds)}", flush=True)

    # --- Load tokenizer from model ---
    m = VoxCPMModel.from_local(BASE, optimize=False, training=False)
    tokenizer = m.text_tokenizer
    del m

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    cols_to_keep = ["feat", "feat_t", "text_ids", "dataset_id"]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text", "split"])
    train_ds = train_ds.select_columns(cols_to_keep)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text", "split"])
    val_ds = val_ds.select_columns(cols_to_keep)

    # --- Save to disk (Arrow format) ---
    os.makedirs(CACHED, exist_ok=True)
    train_path = os.path.join(CACHED, "train")
    val_path = os.path.join(CACHED, "dev")
    print(f"Saving train ({len(train_ds)}) to {train_path} ...", flush=True)
    train_ds.save_to_disk(train_path)
    print(f"Saving dev ({len(val_ds)}) to {val_path} ...", flush=True)
    val_ds.save_to_disk(val_path)
    print(f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
