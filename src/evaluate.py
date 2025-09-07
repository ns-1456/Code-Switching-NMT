#!/usr/bin/env python3
"""
Phase 5b: Evaluation — BLEU, chrF, and Qualitative "Vibe Check" for English → Hinglish NMT.

Runs the model on the full test set and computes:
  - Corpus-level BLEU (via sacrebleu)
  - chrF (better for morphologically rich targets)
  - Side-by-side qualitative report on 20 curated sentences
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from tqdm import tqdm

from src.inference import (
    load_model_for_inference,
    translate_greedy,
    translate_beam,
    batch_translate_greedy,
)
from src.model import load_config


# ======================================================================
# Corpus-level Evaluation
# ======================================================================

def evaluate_test_set(
    model,
    tokenizer,
    config: dict,
    device: torch.device,
