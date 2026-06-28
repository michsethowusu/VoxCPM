#!/usr/bin/env python3
"""Fix max_batch_tokens filtering to handle cached latent dataset (feat_t instead of audio)."""
import math
SCRIPT = "/mnt/volume_d2wey28/projects/voxcpm-ghana/VoxCPM/scripts/train_voxcpm_finetune.py"

with open(SCRIPT, "r") as f:
    content = f.read()

old = '''    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]'''

new = '''    if max_batch_tokens and max_batch_tokens > 0:
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else 0
        # Latent dataset has feat_t directly; no need to decode audio
        if "feat_t" in train_ds.column_names:
            text_lens = [len(t) for t in train_ds["text_ids"]]
            feat_ts = train_ds["feat_t"]
            _patch = base_model.config.patch_size
            est_lengths = [len_t + math.ceil(ft / _patch) + 2 for len_t, ft in zip(text_lens, feat_ts)]
        else:
            from voxcpm.training.data import compute_sample_lengths
            audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
            est_lengths = compute_sample_lengths(
                train_ds,
                audio_vae_fps=audio_vae_fps,
                patch_size=base_model.config.patch_size,
            )
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]'''

assert old in content, "Could not find the max_batch_tokens block"
content = content.replace(old, new, 1)
with open(SCRIPT, "w") as f:
    f.write(content)
print("OK: fixed max_batch_tokens filtering for cached latents")
