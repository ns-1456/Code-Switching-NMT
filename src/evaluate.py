#!/usr/bin/env python3
"""
Phase 5b: Evaluation — BLEU, chrF, and Qualitative "Vibe Check" for English → Hinglish NMT.

Runs the model on the full test set and computes:
  - Corpus-level BLEU (via sacrebleu)
  - chrF (better for morphologically rich targets)
  - Side-by-side qualitative report on 20 curated sentences
"""

from __future__ import annotations

