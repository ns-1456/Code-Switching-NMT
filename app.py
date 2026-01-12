#!/usr/bin/env python3
"""
Phase 6: Streamlit Demo — English → Hinglish Neural Machine Translation.

Features:
  - Text input for English sentence
  - Translate button with greedy/beam search toggle
  - Beam width slider
  - Attention heatmap visualization
  - Clean, modern UI
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

