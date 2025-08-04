#!/usr/bin/env python3
"""
Phase 2: Custom BPE Tokenizer for English → Hinglish NMT.

Trains a Byte-Level BPE tokenizer from scratch on the combined
English + Hinglish training corpus, then wraps it with
PreTrainedTokenizerFast for easy batched encoding/decoding.
"""

from __future__ import annotations

from pathlib import Path

