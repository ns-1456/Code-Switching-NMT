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
