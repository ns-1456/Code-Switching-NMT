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
