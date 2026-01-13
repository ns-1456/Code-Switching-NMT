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

from src.inference import load_model_for_inference, translate_greedy, translate_beam
from src.model import Seq2SeqTransformer


# ======================================================================
# Page Config
# ======================================================================

st.set_page_config(
    page_title="English → Hinglish Translator",
    page_icon="🇮🇳",
    layout="centered",
)


# ======================================================================
# Model Loading (cached)
# ======================================================================

@st.cache_resource
def load_model():
    """Load model once and cache it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model_for_inference(device=device)
    return model, tokenizer, config, device


# ======================================================================
# Attention Heatmap
# ======================================================================

def plot_attention_heatmap(
    attn_weights: list[torch.Tensor],
    src_tokens: list[str],
    tgt_tokens: list[str],
    layer_idx: int = -1,
):
    """
    Plot encoder-decoder cross-attention weights as a heatmap.

    Args:
        attn_weights: list of (batch, heads, tgt_len, src_len) per layer
        src_tokens: source token strings
        tgt_tokens: target token strings
        layer_idx: which layer to visualize (-1 = last)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
