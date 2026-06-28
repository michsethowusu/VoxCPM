"""Build tagged, balanced-split VoxCPM training datasets from our parquet.

- Reads ghana-speech (42 configs) + filtered-English parquet directly (no audio rewrite).
- Prepends a language tag '<|lang:CODE|> ' to each transcript (VoxCPM learns it as text).
- Twi split: Akuapem_Twi -> twi-akuapem, Asante_Twi -> twi-asante.
- Balanced dev split: hold out DEV_PER_LANG clips per language.
- Columns out: audio (Audio 16k), text (tagged), dataset_id (int per language).
"""
import glob, os
from datasets import load_dataset, Audio, concatenate_datasets

GHANA = "/mnt/volume_d2wey28/data/ghana-speech"
ENGLISH = "/mnt/volume_d2wey28/data/ghana-english-tts-filtered/data"

def code_for_dir(dirname):
    if dirname.startswith("Akuapem_Twi"): return "twi-akuapem"
    if dirname.startswith("Asante_Twi"):  return "twi-asante"
    return dirname.split("_")[-1]

def build(dev_per_lang=40, limit_configs=None, limit_rows=None):
    limit_configs = limit_configs or (int(os.environ["GHANA_LIMIT_CONFIGS"]) if os.environ.get("GHANA_LIMIT_CONFIGS") else None)
    limit_rows = limit_rows or (int(os.environ["GHANA_LIMIT_ROWS"]) if os.environ.get("GHANA_LIMIT_ROWS") else None)
    configs = sorted(glob.glob(GHANA + "/*/"))
    if limit_configs:
        configs = configs[:limit_configs]
    # stable dataset_id per language code
    codes = sorted({code_for_dir(os.path.basename(d.rstrip("/"))) for d in configs} | {"en"})
    did = {c: i for i, c in enumerate(codes)}

    trains, vals = [], []
    for d in configs:
        name = os.path.basename(d.rstrip("/"))
        code = code_for_dir(name)
        files = sorted(glob.glob(d + "*.parquet"))
        ds = load_dataset("parquet", data_files=files, split="train")
        if limit_rows:
            ds = ds.select(range(min(limit_rows, len(ds))))
        tag = f"<|lang:{code}|> "
        ds = ds.map(lambda r: {"text": tag + (r["text"] or "").strip(),
                               "dataset_id": did[code]},
                    remove_columns=[c for c in ds.column_names if c not in ("audio", "text")])
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))  # align features before concat
        n_dev = min(dev_per_lang, max(0, len(ds) - 1))
        vals.append(ds.select(range(n_dev)))
        trains.append(ds.select(range(n_dev, len(ds))))

    # English
    ef = sorted(glob.glob(ENGLISH + "/*.parquet"))
    if ef and not limit_configs:
        ed = load_dataset("parquet", data_files=ef, split="train")
        ed = ed.map(lambda r: {"text": "<|lang:en|> " + (r["corrected_text"] or "").strip(),
                               "dataset_id": did["en"]},
                    remove_columns=[c for c in ed.column_names if c not in ("audio",)])
        ed = ed.cast_column("audio", Audio(sampling_rate=16000))
        vals.append(ed.select(range(dev_per_lang)))
        trains.append(ed.select(range(dev_per_lang, len(ed))))

    train = concatenate_datasets(trains)
    val = concatenate_datasets(vals)
    return train, val, did

if __name__ == "__main__":
    # quick validation on 2 small configs, few rows
    tr, va, did = build(dev_per_lang=5, limit_configs=2, limit_rows=30)
    print("train", len(tr), "val", len(va))
    print("sample text:", repr(tr[0]["text"])[:80])
    print("sample dataset_id:", tr[0]["dataset_id"])
    print("audio sr:", tr[0]["audio"]["sampling_rate"], "len:", len(tr[0]["audio"]["array"]))
