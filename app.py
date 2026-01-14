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
    import seaborn as sns

    if not attn_weights:
        st.warning("No attention weights available.")
        return

    # Get the specified layer's attention, averaged across heads
    attn = attn_weights[layer_idx]  # (1, heads, tgt_len, src_len)
    attn = attn.squeeze(0).mean(dim=0).cpu().numpy()  # (tgt_len, src_len)

    # Trim to actual token lengths
    tgt_len = min(len(tgt_tokens), attn.shape[0])
    src_len = min(len(src_tokens), attn.shape[1])
    attn = attn[:tgt_len, :src_len]

    fig, ax = plt.subplots(figsize=(max(8, src_len * 0.6), max(4, tgt_len * 0.5)))

    sns.heatmap(
        attn,
        xticklabels=src_tokens[:src_len],
        yticklabels=tgt_tokens[:tgt_len],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_xlabel("Source (English)", fontsize=12)
    ax.set_ylabel("Target (Hinglish)", fontsize=12)
    ax.set_title("Cross-Attention Heatmap (Averaged over Heads)", fontsize=13)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ======================================================================
# UI
# ======================================================================

def main():
    # Header
    st.title("English → Hinglish Translator")
