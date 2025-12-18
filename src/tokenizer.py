#!/usr/bin/env python3
"""
Phase 2: Custom BPE Tokenizer for English → Hinglish NMT.

Trains a Byte-Level BPE tokenizer from scratch on the combined
English + Hinglish training corpus, then wraps it with
PreTrainedTokenizerFast for easy batched encoding/decoding.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_corpus(data_dir: Path) -> Path:
    """
    Concatenate en + hi_ng columns from train.csv into a single
    corpus.txt file for tokenizer training.
    """
    import pandas as pd

    train_df = pd.read_csv(data_dir / "train.csv")
    corpus_path = data_dir / "corpus.txt"

    with open(corpus_path, "w", encoding="utf-8") as f:
