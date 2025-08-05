#!/usr/bin/env python3
"""
Phase 3: From-Scratch Encoder-Decoder Transformer for English → Hinglish NMT.

Every component is hand-written in PyTorch:
  - Sinusoidal Positional Encoding
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Position-wise Feed-Forward Network
  - Encoder Layer / Decoder Layer (Pre-Norm)
  - Full Encoder / Decoder stacks
  - Seq2SeqTransformer wrapper with output projection
  - Mask utilities (padding + causal)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ======================================================================
# Configuration
# ======================================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
