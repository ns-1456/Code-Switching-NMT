#!/usr/bin/env python3
"""
Phase 4: Training Loop for English → Hinglish NMT.

Implements:
  - TranslationDataset (torch Dataset from CSV)
  - Collate function with dynamic padding
  - Training loop with teacher forcing
  - AdamW optimizer with warmup + cosine decay
  - Label smoothing via CrossEntropyLoss
  - Gradient clipping
  - Early stopping on validation loss
  - Qualitative monitor (translate fixed sentences every epoch)
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import (
    Seq2SeqTransformer,
    build_model,
    create_padding_mask,
    create_tgt_mask,
