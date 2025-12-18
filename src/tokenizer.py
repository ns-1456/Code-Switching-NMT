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
        for _, row in train_df.iterrows():
            f.write(str(row["en"]) + "\n")
            f.write(str(row["hi_ng"]) + "\n")

    print(f"[tokenizer] Wrote corpus ({len(train_df) * 2:,} lines) to {corpus_path}")
    return corpus_path


def train_tokenizer(config: dict | None = None) -> PreTrainedTokenizerFast:
    """
    Train a ByteLevelBPE tokenizer from scratch and save it.
    Returns the wrapped PreTrainedTokenizerFast.
    """
    if config is None:
        config = load_config()

    tok_cfg = config["tokenizer"]
    data_dir = Path(config["data"]["data_dir"])
    save_dir = Path(tok_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
