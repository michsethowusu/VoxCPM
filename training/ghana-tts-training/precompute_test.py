import io, glob, numpy as np, soundfile as sf, torch
from voxcpm.model.voxcpm import VoxCPMModel

BASE = "/mnt/volume_d2wey28/models/VoxCPM-0.5B"
m = VoxCPMModel.from_local(BASE, optimize=False, training=False)
vae = m.audio_vae.to("cuda").eval()
SR = vae.sample_rate
patch_len = vae.hop_length * 1  # patch_size handled downstream; align to hop
print("VAE sample_rate", SR, "hop_length", vae.hop_length)

import pyarrow.parquet as pq
f = sorted(glob.glob("/mnt/volume_d2wey28/data/ghana-speech/Anyin_any/*.parquet"))[0]
rows = pq.ParquetFile(f).read_row_groups([0]).to_pylist()[:3]
for r in rows:
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    wav = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to("cuda")  # [1,1,T]
    if wav.size(-1) % patch_len: wav = torch.nn.functional.pad(wav, (0, patch_len - wav.size(-1) % patch_len))
    with torch.no_grad():
        z = vae.encode(wav, SR)            # [1, D, T']
        feat = z.transpose(1, 2).squeeze(0)  # [T', D]
    print(f"dur={len(a)/sr:.1f}s -> feat {tuple(feat.shape)} dtype {feat.dtype}  fp16 bytes={feat.half().cpu().numpy().nbytes}")
