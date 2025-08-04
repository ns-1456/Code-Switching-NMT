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

    # Build corpus.txt from training data
    corpus_path = build_corpus(data_dir)

    # Special tokens (order determines IDs: pad=0, sos=1, eos=2, unk=3)
    special_tokens = [
        tok_cfg["special_tokens"]["pad"],  # 0
        tok_cfg["special_tokens"]["sos"],  # 1
        tok_cfg["special_tokens"]["eos"],  # 2
        tok_cfg["special_tokens"]["unk"],  # 3
    ]

    # Train BPE
    print(f"[tokenizer] Training ByteLevelBPE (vocab_size={tok_cfg['vocab_size']}) ...")
    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[str(corpus_path)],
        vocab_size=tok_cfg["vocab_size"],
        min_frequency=tok_cfg["min_frequency"],
        special_tokens=special_tokens,
    )

    # Save raw tokenizer files
    bpe.save_model(str(save_dir))
    print(f"[tokenizer] Saved vocab.json + merges.txt to {save_dir}/")

    # Wrap with PreTrainedTokenizerFast
    tokenizer = _load_fast_tokenizer(save_dir, tok_cfg)
    print(f"[tokenizer] Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def load_tokenizer(config: dict | None = None) -> PreTrainedTokenizerFast:
    """Load a previously trained tokenizer from disk."""
    if config is None:
        config = load_config()

    tok_cfg = config["tokenizer"]
    save_dir = Path(tok_cfg["save_dir"])

    if not (save_dir / "vocab.json").exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {save_dir}. Run train_tokenizer() first."
        )

    return _load_fast_tokenizer(save_dir, tok_cfg)


def _load_fast_tokenizer(
    save_dir: Path, tok_cfg: dict
) -> PreTrainedTokenizerFast:
    """Wrap the ByteLevelBPE files in a PreTrainedTokenizerFast."""
    vocab_path = save_dir / "vocab.json"
    merges_path = save_dir / "merges.txt"

    backend = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=tok_cfg["special_tokens"]["pad"],
        bos_token=tok_cfg["special_tokens"]["sos"],
        eos_token=tok_cfg["special_tokens"]["eos"],
        unk_token=tok_cfg["special_tokens"]["unk"],
    )
    return tokenizer


def encode(text: str, tokenizer: PreTrainedTokenizerFast) -> list[int]:
