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
    method: str = "greedy",
    beam_width: int = 5,
    batch_size: int = 128,
) -> dict:
    """
    Evaluate the model on the full test set.

    Uses batched greedy decoding by default (~50-100x faster than
    one-by-one beam search). Beam search is better reserved for the
    vibe check where quality matters on a few sentences.

    Returns dict with:
      - bleu: corpus BLEU score
      - chrf: chrF score
      - predictions: list of (source, prediction, reference)
    """
    data_dir = Path(config["data"]["data_dir"])
    test_df = pd.read_csv(data_dir / "test.csv")

    sources = [str(s) for s in test_df["en"].tolist()]
    references = [str(s) for s in test_df["hi_ng"].tolist()]

