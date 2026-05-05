"""HDF5-backed PyTorch Dataset for GenFic register training.

Loads from `source/{register}.h5` produced by `scripts/build_dataset.py`.
One sample = one prompt-and-response pair packed to `seq_len` tokens.

The H5 is expected to carry:
- `input_ids`        int32  [N, L]   pad-token-padded
- `attention_mask`   int8   [N, L]
- `prompt_lengths`   int16  [N]      length of the `[INST] ... [/INST] ` prefix
- `split`            int8   [N]      0=train, 1=val, 2=test
- attrs: `pad_token_id`, `eos_token_id`, `seq_len`, `register`

Loss is computed only on the response (chapter chunk + EOS); the prompt prefix
and pad positions are masked with `LOSS_IGNORE_INDEX = -100`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

LOSS_IGNORE_INDEX = -100


class GenFicDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",  # "train" | "val" | "test"
    ):
        self.h5_path = str(h5_path)
        self.split = split
        with h5py.File(self.h5_path, "r") as f:
            split_arr = f["split"][:]
            self.pad_token_id = int(f.attrs["pad_token_id"])
            self.eos_token_id = int(f.attrs["eos_token_id"])
            self.seq_len = int(f.attrs["seq_len"])
            self.register = str(f.attrs.get("register", "unknown"))
            self.authors = list(f["authors"].asstr()) if "authors" in f else []
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.indices = np.where(split_arr == split_id)[0]
        self._h5 = None  # lazy-opened per worker

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_open()
        i = int(self.indices[idx])
        input_ids = torch.from_numpy(self._h5["input_ids"][i].astype(np.int64))
        attention_mask = torch.from_numpy(self._h5["attention_mask"][i].astype(np.int64))
        prompt_len = int(self._h5["prompt_lengths"][i])

        labels = input_ids.clone()
        labels[attention_mask == 0] = LOSS_IGNORE_INDEX
        if prompt_len > 0:
            labels[:prompt_len] = LOSS_IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
