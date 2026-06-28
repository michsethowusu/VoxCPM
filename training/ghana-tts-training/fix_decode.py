#!/usr/bin/env python3
"""Fix tokenizer.decode -> tokenizer.tokenizer.decode in the cached dataset path."""
SCRIPT = "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/scripts/train_voxcpm_finetune.py"

with open(SCRIPT, "r") as f:
    content = f.read()

old = 'val_texts = [tokenizer.decode(list(t), skip_special_tokens=False) for t in _raw]'
new = 'val_texts = [tokenizer.tokenizer.decode(list(t), skip_special_tokens=False) for t in _raw]'

assert old in content, "Could not find the decode line"
content = content.replace(old, new, 1)
with open(SCRIPT, "w") as f:
    f.write(content)
print("OK: fixed tokenizer.decode -> tokenizer.tokenizer.decode")
