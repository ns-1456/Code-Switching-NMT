#!/usr/bin/env python3
"""
Phase 1: Data Pipeline for English → Hinglish NMT.

Downloads findnitai/english-to-hinglish from HuggingFace, cleans it
(Devanagari filter, length filter, normalization, dedup), splits 90/5/5,
and saves train/val/test CSVs.
"""

from __future__ import annotations

import re
