#!/usr/bin/env python3
"""
Phase 1: Data Pipeline for English → Hinglish NMT.

Downloads findnitai/english-to-hinglish from HuggingFace, cleans it
(Devanagari filter, length filter, normalization, dedup), splits 90/5/5,
and saves train/val/test CSVs.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Regex patterns for cleaning
# ---------------------------------------------------------------------------
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
# Broad emoji pattern (covers most emoji blocks)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "]+",
    flags=re.UNICODE,
)
MULTI_SPACE_RE = re.compile(r"\s+")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Lowercase, strip URLs/handles/hashtags/emojis, normalize whitespace."""
    text = text.lower()
    text = URL_RE.sub("", text)
    text = HANDLE_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    # Normalize unicode (e.g. combining characters)
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    """Count words by splitting on whitespace."""
    return len(text.split())


def run_pipeline(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full data pipeline: download, clean, split, save.

    Returns (train_df, val_df, test_df).
    """
    if config is None:
        config = load_config()

    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download dataset from HuggingFace
    # ------------------------------------------------------------------
    from datasets import load_dataset

    print("[data] Downloading findnitai/english-to-hinglish ...")
    ds = load_dataset(data_cfg["dataset_name"], split="train")
    df = ds.to_pandas()
    print(f"[data] Raw rows: {len(df):,}")

    en_col = data_cfg["en_col"]
    hi_col = data_cfg["hi_col"]

    # Keep only the columns we need
    df = df[[en_col, hi_col]].copy()
    df.columns = ["en", "hi_ng"]

    # ------------------------------------------------------------------
    # 2. Drop nulls / empty
    # ------------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["en", "hi_ng"])
    df = df[(df["en"].str.strip() != "") & (df["hi_ng"].str.strip() != "")]
    print(f"[data] After null/empty filter: {len(df):,} (dropped {before - len(df):,})")

    # ------------------------------------------------------------------
    # 3. Devanagari script filter
    # ------------------------------------------------------------------
    before = len(df)
    mask = df["hi_ng"].apply(lambda x: bool(DEVANAGARI_RE.search(str(x))))
    df = df[~mask]
    print(f"[data] After Devanagari filter: {len(df):,} (dropped {before - len(df):,})")

    # ------------------------------------------------------------------
    # 4. Normalize text
    # ------------------------------------------------------------------
    df["en"] = df["en"].apply(normalize_text)
    df["hi_ng"] = df["hi_ng"].apply(normalize_text)

    # Drop rows that became empty after normalization
    df = df[(df["en"].str.strip() != "") & (df["hi_ng"].str.strip() != "")]

    # ------------------------------------------------------------------
