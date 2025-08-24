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

        # Encode source
        enc_output = model.encode(src, src_mask)

        # Greedy decode
        tgt_ids = [sos_id]
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            tgt_mask = create_tgt_mask(tgt, pad_idx)

            dec_output = model.decode(tgt, enc_output, tgt_mask, src_mask)
            logits = model.generator(dec_output[:, -1, :])
            next_token = logits.argmax(dim=-1).item()

            if next_token == eos_id:
                break
            tgt_ids.append(next_token)

        # Decode (skip <sos>)
        output = tokenizer.decode(tgt_ids[1:], skip_special_tokens=True)
        print(f"    EN: {sent}")
        print(f"    HI: {output}")
        print()

    model.train()


# ======================================================================
# Training Function
# ======================================================================

def train(config: dict | None = None):
    """Full training pipeline: load data, build model, train, save."""

    if config is None:
        config = load_config()

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(config)
    pad_idx = tokenizer.pad_token_id  # 0
    sos_id = tokenizer.bos_token_id   # 1
    eos_id = tokenizer.eos_token_id   # 2
    vocab_size = tokenizer.vocab_size

    print(f"[train] Vocab size: {vocab_size}")

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    data_dir = Path(data_cfg["data_dir"])

    train_ds = TranslationDataset(
        data_dir / "train.csv", tokenizer, max_len=model_cfg["max_seq_len"],
        sos_id=sos_id, eos_id=eos_id,
    )
    val_ds = TranslationDataset(
        data_dir / "val.csv", tokenizer, max_len=model_cfg["max_seq_len"],
        sos_id=sos_id, eos_id=eos_id,
    )

    collate = lambda batch: collate_fn(batch, pad_idx=pad_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model(config, vocab_size=vocab_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Optimizer + Scheduler + Loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * train_cfg["num_epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=total_steps,
        max_lr=train_cfg["learning_rate"],
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=train_cfg["label_smoothing"],
    )

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    monitor_sentences = eval_cfg["monitor_sentences"]

    print(f"[train] Starting training for {train_cfg['num_epochs']} epochs ...")
    print(f"[train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['num_epochs']} [Train]")
        for src, tgt in pbar:
            src = src.to(device)  # (batch, src_len)
            tgt = tgt.to(device)  # (batch, tgt_len)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Create masks
            src_mask = create_padding_mask(src, pad_idx)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx)

            # Forward
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Reshape for loss: (batch * tgt_len, vocab_size) vs (batch * tgt_len,)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_lr():.2e}")

        avg_train_loss = train_loss / max(1, num_batches)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch}/{train_cfg['num_epochs']} [Val]"):
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = create_padding_mask(src, pad_idx)
                tgt_mask = create_tgt_mask(tgt_input, pad_idx)

                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1),
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        elapsed = time.time() - epoch_start

        print(
            f"\n  Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  time={elapsed:.1f}s  "
            f"lr={scheduler.get_lr():.2e}"
        )

        # --- Qualitative check ---
        qualitative_check(
            model, tokenizer, monitor_sentences, device,
            max_len=model_cfg["max_seq_len"],
            sos_id=sos_id, eos_id=eos_id, pad_idx=pad_idx,
        )

        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  [checkpoint] Saved best model (val_loss={avg_val_loss:.4f}) to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  [early-stop] No improvement. Patience: {patience_counter}/{train_cfg['early_stopping_patience']}")

            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"  [early-stop] Stopping training at epoch {epoch}.")
                break

    print(f"\n[train] Training complete. Best val_loss: {best_val_loss:.4f}")
    return model


# ======================================================================
# Main
# ======================================================================
