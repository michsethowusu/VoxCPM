import os, time, glob, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi

REPO = "ghananlpcommunity/voxcpm-ghana-latents"
SRC = "/mnt/volume_d2wey28/data/voxcpm-latents"
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
print(f"[push] {REPO} (public)", flush=True)

# derive dataset_id map exactly as the precompute did (sorted unique code prefixes)
codes = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(SRC + "/*.parquet")})
did = {c: i for i, c in enumerate(codes)}
did_md = "\n".join(f"| `{c}` | {i} |" for c, i in did.items())

CARD = f"""---
license: cc-by-4.0
task_categories:
- text-to-speech
language:
- en
pretty_name: VoxCPM Ghana — Precomputed AudioVAE Latents
size_categories:
- 1M<n<10M
---

# VoxCPM Ghana — Precomputed AudioVAE Latents

The **exact training-ready data** used to fine-tune
[`ghananlpcommunity/voxcpm-ghana`](https://huggingface.co/ghananlpcommunity/voxcpm-ghana):
precomputed **VoxCPM-0.5B AudioVAE latents (16 kHz)** for 42 Ghanaian languages +
filtered Ghanaian English, with **language-tagged** transcripts. Drop-in for
VoxCPM fine-tuning — no audio decoding or VAE encoding needed at train time.

- **1,756,157 clips · ~3,400 h · 16 kHz**
- 42 Ghanaian languages (incl. **Twi split**: `twi-asante`, `twi-akuapem`) + `en`
- AudioVAE from [`openbmb/VoxCPM-0.5B`](https://huggingface.co/openbmb/VoxCPM-0.5B): **64-dim, hop 640 (~25 fps)**

## Format (parquet shards, `CODE__*.parquet`)
| column | type | notes |
|---|---|---|
| `feat` | binary | fp16 AudioVAE latent, shape `[feat_t, 64]`. Reconstruct: `np.frombuffer(feat, np.float16).reshape(feat_t, 64)` |
| `feat_t` | int32 | number of latent frames (~25 per second) |
| `text` | string | transcript **prefixed with the language tag** `<|lang:CODE|> ` |
| `dataset_id` | int32 | per-language id (table below) |
| `split` | string | `train` / `dev` (balanced: ~40 dev clips per language) |

## Language tags
Each transcript starts with `<|lang:CODE|> `, where `CODE` is the ISO-639-3 code
(e.g. `ewe`, `dag`, `hau`, `fat`), with Twi split into `twi-asante` / `twi-akuapem`,
and `en` for English. The model learns the tag as text (VoxCPM is tokenizer-free).

## dataset_id map
| code | id |
|---|---|
{did_md}

## Source
Derived from [`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
and [`ghananlpcommunity/ghana-english-tts-filtered`](https://huggingface.co/datasets/ghananlpcommunity/ghana-english-tts-filtered).
"""
open("/tmp/_README_lat.md", "w").write(CARD)

for att in range(1, 16):
    try:
        print(f"[push] attempt {att}", flush=True)
        api.upload_file(path_or_fileobj="/tmp/_README_lat.md", path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
        api.upload_folder(folder_path=SRC, repo_id=REPO, repo_type="dataset", commit_message="Add precomputed VoxCPM latents")
        print("PUSH_LATENTS_DONE", flush=True); break
    except Exception as e:
        print(f"[push] retry {att}: {type(e).__name__}: {str(e)[:120]}", flush=True); time.sleep(15)
