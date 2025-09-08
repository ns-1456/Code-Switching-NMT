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

    print(f"[eval] Evaluating on {len(test_df):,} test examples ({method} decoding) ...")

    if method == "greedy":
        # ---- Fast batched greedy (GPU-parallel) ----
        num_batches = (len(sources) + batch_size - 1) // batch_size
        print(f"[eval] Using batched greedy: {num_batches} batches of {batch_size}")
        predictions = batch_translate_greedy(
            model, tokenizer, sources, device, batch_size=batch_size,
        )
    else:
        # ---- One-by-one beam search (slow, use only if needed) ----
        predictions = []
        for idx in tqdm(range(len(sources)), desc="Evaluating (beam)"):
            pred_text, _ = translate_beam(
                model, tokenizer, sources[idx], device, beam_width=beam_width
            )
            predictions.append(pred_text)

    # Compute BLEU (sacrebleu expects a list of reference lists)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    print(f"\n[eval] Results:")
    print(f"  BLEU:  {bleu.score:.2f}")
    print(f"  chrF:  {chrf.score:.2f}")

    return {
        "bleu": bleu,
        "chrf": chrf,
        "predictions": list(zip(sources, predictions, references)),
    }


# ======================================================================
# Qualitative "Vibe Check"
# ======================================================================

VIBE_CHECK_SENTENCES = [
    "I am going home",
    "Are you crazy?",
    "Let's meet tomorrow",
    "What is your name?",
    "I don't understand this",
    "Please give me some water",
    "Where is the nearest hospital?",
