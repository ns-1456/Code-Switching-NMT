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
