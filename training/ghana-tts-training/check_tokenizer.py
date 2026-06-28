#!/usr/bin/env python3
import sys
sys.path.insert(0, "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/src")
from voxcpm.model.voxcpm import VoxCPMModel
import warnings
warnings.filterwarnings("ignore")

m = VoxCPMModel.from_local("/mnt/volume_d2wey28/models/VoxCPM-0.5B", optimize=False, training=False)
tok = m.text_tokenizer
print(f"type: {type(tok)}")
print(f"MRO: {[c.__name__ for c in type(tok).__mro__]}")
print(f"has decode: {hasattr(tok, 'decode')}")

# Check wrapped tokenizer
for attr in ["tokenizer", "_tokenizer", "hf_tokenizer", "sp_model", "vocab"]:
    if hasattr(tok, attr):
        obj = getattr(tok, attr)
        print(f"has .{attr}: type={type(obj)}")
        if hasattr(obj, "decode"):
            print(f"  .{attr}.decode exists!")
        break

# Try direct call
try:
    result = tok("hello world")
    print(f"tok('hello'): {result}")
except Exception as e:
    print(f"tok('hello') error: {e}")
