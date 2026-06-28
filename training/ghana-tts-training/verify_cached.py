#!/usr/bin/env python3
import sys
sys.path.insert(0, "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/src")
from datasets import load_from_disk
from voxcpm.model.voxcpm import VoxCPMModel

ds = load_from_disk("/mnt/volume_d2wey28/data/voxcpm-latents-cached/train")
print(f"train samples: {len(ds)}")
print(f"columns: {ds.column_names}")

m = VoxCPMModel.from_local("/mnt/volume_d2wey28/models/VoxCPM-0.5B", optimize=False, training=False)
tok = m.text_tokenizer

for i in range(5):
    item = ds[i]
    txt = tok.decode(list(item["text_ids"]))
    did = item["dataset_id"]
    ft = item["feat_t"]
    print(f"[{i}] dataset_id={did} feat_t={ft} text={txt[:120]}")
