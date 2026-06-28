import math
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_REF_AUDIO_COLUMN = "ref_audio"
DEFAULT_ID_COLUMN = "dataset_id"


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    ref_audio_column: str = DEFAULT_REF_AUDIO_COLUMN,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)

        # ref_audio is optional — cast to Audio if the column exists
        ref_col = ref_audio_column if ref_audio_column in ds.column_names else DEFAULT_REF_AUDIO_COLUMN
        if ref_col in ds.column_names:
            ds = ds.cast_column(ref_col, Audio(sampling_rate=sample_rate))
            if ref_col != DEFAULT_REF_AUDIO_COLUMN:
                ds = ds.rename_column(ref_col, DEFAULT_REF_AUDIO_COLUMN)

        if dataset_id_column and dataset_id_column in ds.column_names:
            if dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: int = 25,
    patch_size: int = 1,
) -> List[int]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker / AudioVAE 一致：
    - 文本长度: len(text_ids)
    - 音频长度:
        duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae
        t_seq = ceil(t_vae / patch_size)
    - 无 ref_audio: text_len + t_seq + 2
    - 有 ref_audio: text_len + t_seq + ref_seq + 4

    Optimized: Use batch column access instead of iterating item by item.
    """
    text_ids_list = ds["text_ids"]
    text_lens = [len(t) for t in text_ids_list]

    has_duration = "duration" in ds.column_names
    if has_duration:
        durations = ds["duration"]
    else:
        durations = []
        for i in range(len(ds)):
            audio = ds[i][DEFAULT_AUDIO_COLUMN]
            durations.append(len(audio["array"]) / float(audio["sampling_rate"]))

    has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in ds.column_names
    if has_ref_audio:
        ref_duration_col = "ref_duration" if "ref_duration" in ds.column_names else None

    lengths = []
    for i, (text_len, duration) in enumerate(zip(text_lens, durations)):
        t_vae = math.ceil(float(duration) * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)

        ref_seq = 0
        if has_ref_audio:
            # Estimate ref_audio length; ref_audio is None for samples without it
            if ref_duration_col:
                ref_dur = ds[i].get(ref_duration_col)
            else:
                ref_item = ds[i].get(DEFAULT_REF_AUDIO_COLUMN)
                ref_dur = len(ref_item["array"]) / float(ref_item["sampling_rate"]) if ref_item else None
            if ref_dur is not None and float(ref_dur) > 0:
                ref_vae = math.ceil(float(ref_dur) * audio_vae_fps)
                ref_seq = math.ceil(ref_vae / patch_size)

        # +2 for 101/102; +2 more for 103/104 when ref_audio present
        overhead = 4 if ref_seq > 0 else 2
        total_len = text_len + t_seq + ref_seq + overhead
        lengths.append(total_len)

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    Thin wrapper around a tokenized HuggingFace dataset that returns
    PyTorch-friendly samples.
    """

    _SENTINEL = [-100.0]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in dataset.column_names
        # precomputed-latent mode: dataset carries fp16 VAE feats instead of audio
        self.is_latent = "feat" in dataset.column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if self.is_latent:
            import numpy as np
            feat = np.frombuffer(item["feat"], dtype=np.float16).reshape(int(item["feat_t"]), -1)
            return {
                "text_ids": item["text_ids"],
                "audio_array": feat.astype(np.float32),  # 2D [T', D] latent
                "audio_sampling_rate": 16000,
                "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
                "is_prompt": item.get("is_prompt", False),
            }
        audio = item[DEFAULT_AUDIO_COLUMN]
        sample = {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }
        if self.has_ref_audio:
            ref = item.get(DEFAULT_REF_AUDIO_COLUMN)
            sample["ref_audio_array"] = ref["array"] if ref else self._SENTINEL
        return sample

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        import numpy as np
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]
        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        # Precomputed latents ([T', D]) are passed as a LIST (exact length, no padding);
        # raw waveforms (1D) are padded into a tensor as before.
        is_latent = np.asarray(batch[0]["audio_array"]).ndim == 2
        if is_latent:
            audio_tokens = [torch.as_tensor(np.asarray(s["audio_array"]), dtype=torch.float32) for s in batch]
        else:
            audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
            audio_tokens = cls.pad_sequences(audio_tensors, pad_value=-100.0)

        result = {
            "text_tokens": text_padded,
            "audio_tokens": audio_tokens,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }

        if "ref_audio_array" in batch[0]:
            ref_tensors = [torch.tensor(s["ref_audio_array"], dtype=torch.float32) for s in batch]
            result["ref_audio_tokens"] = cls.pad_sequences(ref_tensors, pad_value=-100.0)

        return result


class _LatentVAEShim:
    """Stand-in for AudioVAE used only to construct the packer inside DataLoader
    workers for the precomputed-latent path. That path never calls the VAE
    (``encode_audio`` passes 2D feats straight through and ``patch_len`` is
    unused), so these values are inert; they mirror VoxCPM-0.5B (hop=640, 16kHz)
    for completeness."""

    hop_length = 640
    sample_rate = 16_000


# Per-worker-process lazy packer singleton (avoids pickling the packer; only the
# small config tuple crosses the process boundary inside _LatentPackCollate).
_LATENT_PACKER = None
_LATENT_PACKER_KEY = None


def _get_latent_packer(dataset_cnt: int, max_len: int, patch_size: int, feat_dim: int):
    global _LATENT_PACKER, _LATENT_PACKER_KEY
    key = (dataset_cnt, max_len, patch_size, feat_dim)
    if _LATENT_PACKER is None or _LATENT_PACKER_KEY != key:
        _LATENT_PACKER = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=max_len,
            patch_size=patch_size,
            feat_dim=feat_dim,
            audio_vae=_LatentVAEShim(),
        )
        _LATENT_PACKER_KEY = key
    return _LATENT_PACKER


class _LatentPackCollate:
    """Collate that builds the FULL packed batch on CPU inside each DataLoader
    worker — parallelised across workers and overlapped with GPU compute via
    prefetch. The main process then only transfers the packed tensors to GPU.

    This replaces the previous flow where packing ran serially in the main
    process on the GPU: ~1k tiny CUDA kernels per step (one per sample × per
    tensor op) plus host syncs starved the GPU to 0-3% utilisation. The packer
    is pure tensor reshaping/masking, so running it on CPU is bit-identical and
    far cheaper (no kernel-launch or sync overhead). Picklable (only ints)."""

    def __init__(self, dataset_cnt: int, max_len: int, patch_size: int, feat_dim: int):
        self.cfg = (int(dataset_cnt), int(max_len), int(patch_size), int(feat_dim))

    def __call__(self, batch: List[Dict]):
        base = HFVoxCPMDataset.collate_fn(batch)
        packer = _get_latent_packer(*self.cfg)
        return packer(
            audio_tokens=base["audio_tokens"],
            text_tokens=base["text_tokens"],
            task_ids=base["task_ids"],
            dataset_ids=base["dataset_ids"],
            is_prompts=base["is_prompts"],
            ref_audio_tokens=base.get("ref_audio_tokens"),
        )


class BatchProcessor:
    """
    Wraps ``AudioFeatureProcessingPacker`` so the training loop can mirror
    the minicpm-audio mechanics.
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Latent path: the batch was already packed in the DataLoader workers
        # (see _LatentPackCollate). Just move the packed tensors to the device.
        if "audio_feats" in batch:
            return {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        _at = batch["audio_tokens"]
        audio_tokens = [t.to(self.device) for t in _at] if isinstance(_at, list) else _at.to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)

        ref_audio_tokens = None
        if "ref_audio_tokens" in batch:
            ref_audio_tokens = batch["ref_audio_tokens"].to(self.device)

        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
            ref_audio_tokens=ref_audio_tokens,
        )
        return packed


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
    pack_config: Optional[dict] = None,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    # Precomputed-latent path: pack the batch inside the workers (parallel +
    # overlapped) instead of serially on the GPU in the main process.
    if torch_dataset.is_latent and pack_config is not None:
        collate_fn = _LatentPackCollate(**pack_config)
    else:
        collate_fn = HFVoxCPMDataset.collate_fn
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
