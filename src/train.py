#!/usr/bin/env python3
"""
Phase 4: Training Loop for English → Hinglish NMT.

Implements:
  - TranslationDataset (torch Dataset from CSV)
  - Collate function with dynamic padding
  - Training loop with teacher forcing
  - AdamW optimizer with warmup + cosine decay
  - Label smoothing via CrossEntropyLoss
  - Gradient clipping
  - Early stopping on validation loss
  - Qualitative monitor (translate fixed sentences every epoch)
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import (
    Seq2SeqTransformer,
    build_model,
    create_padding_mask,
    create_tgt_mask,
    load_config,
)
from src.tokenizer import load_tokenizer


# ======================================================================
# Dataset
# ======================================================================

class TranslationDataset(Dataset):
    """
    Reads a CSV with 'en' and 'hi_ng' columns and tokenizes on-the-fly.
    Each item returns (src_ids, tgt_ids) as 1-D LongTensors.
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        max_len: int = 64,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        en_text = str(row["en"])
        hi_text = str(row["hi_ng"])

        # Encode (without special tokens — we add them manually)
        src_ids = self.tokenizer.encode(en_text, add_special_tokens=False)
        tgt_ids = self.tokenizer.encode(hi_text, add_special_tokens=False)

        # Truncate to max_len - 2 (room for <sos> and <eos>)
        src_ids = src_ids[: self.max_len - 2]
        tgt_ids = tgt_ids[: self.max_len - 2]

        # Add <sos> and <eos>
        src_ids = [self.sos_id] + src_ids + [self.eos_id]
        tgt_ids = [self.sos_id] + tgt_ids + [self.eos_id]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# ======================================================================
# Collate Function
# ======================================================================

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 0):
    """
    Pads src and tgt sequences to the max length in the batch.

    Returns:
        src: (batch, max_src_len)
        tgt: (batch, max_tgt_len)
    """
    src_seqs, tgt_seqs = zip(*batch)

    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)

    return src_padded, tgt_padded


# ======================================================================
# Learning Rate Scheduler (Warmup + Cosine Decay)
# ======================================================================

class WarmupCosineScheduler:
    """
    Warmup for `warmup_steps`, then cosine-decay to `min_lr`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def get_lr(self) -> float:
        return self._compute_lr()


# ======================================================================
# Qualitative Monitor
# ======================================================================

def qualitative_check(
    model: Seq2SeqTransformer,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    max_len: int = 64,
    sos_id: int = 1,
    eos_id: int = 2,
    pad_idx: int = 0,
):
    """
    Translate a fixed set of sentences using greedy decoding and print results.
    Used to monitor training quality every epoch.
    """
    model.eval()
    print("\n  --- Qualitative Check ---")

    for sent in sentences:
        src_ids = tokenizer.encode(sent, add_special_tokens=False)
        src_ids = [sos_id] + src_ids + [eos_id]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = create_padding_mask(src, pad_idx)

