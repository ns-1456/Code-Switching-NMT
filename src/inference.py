#!/usr/bin/env python3
"""
Phase 5a: Inference — Greedy Decoding and Beam Search for English → Hinglish NMT.

Provides:
  - translate_greedy: simple argmax decoding (fast, for debugging)
  - translate_beam: beam search decoding (better quality, for demo/eval)
  - load_model_for_inference: loads checkpoint + tokenizer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from src.model import (
    Seq2SeqTransformer,
    build_model,
    create_padding_mask,
    create_tgt_mask,
    load_config,
)
from src.tokenizer import load_tokenizer


# ======================================================================
# Model Loading
# ======================================================================

def load_model_for_inference(
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "configs/config.yaml",
    device: torch.device | None = None,
) -> tuple[Seq2SeqTransformer, object, dict]:
    """
    Load a trained model from checkpoint.

    Returns:
        (model, tokenizer, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    tokenizer = load_tokenizer(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(config, vocab_size=tokenizer.vocab_size)

    # Filter out positional encoding buffers from checkpoint — they may
    # have been saved with a smaller size (e.g. 64) than the current model
    # uses (512). The PE buffer is deterministic (sinusoidal), so the fresh
    # one in the new model is identical for overlapping positions.
    saved_state = checkpoint["model_state_dict"]
    pe_keys = [k for k in saved_state if ".pe" in k]
    for k in pe_keys:
        if saved_state[k].shape != model.state_dict()[k].shape:
            del saved_state[k]

    model.load_state_dict(saved_state, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[inference] Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
