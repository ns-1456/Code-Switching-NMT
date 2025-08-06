#!/usr/bin/env python3
"""
Phase 3: From-Scratch Encoder-Decoder Transformer for English → Hinglish NMT.

Every component is hand-written in PyTorch:
  - Sinusoidal Positional Encoding
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Position-wise Feed-Forward Network
  - Encoder Layer / Decoder Layer (Pre-Norm)
  - Full Encoder / Decoder stacks
  - Seq2SeqTransformer wrapper with output projection
  - Mask utilities (padding + causal)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ======================================================================
# Configuration
# ======================================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# Positional Encoding (Sinusoidal — Vaswani et al.)
# ======================================================================

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional embeddings to token embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Dynamically extends the buffer if the input sequence is longer than
    the pre-allocated buffer (handles any checkpoint / config mismatch).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build the sinusoidal PE table of shape (1, max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        seq_len = x.size(1)

        # Dynamically extend buffer if input is longer than current PE table
        if seq_len > self.pe.size(1):
            self.pe = self._build_pe(seq_len + 64, self.d_model).to(x.device)

        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ======================================================================
# Scaled Dot-Product Attention
# ======================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  broadcastable to (batch, heads, seq_q, seq_k), True = masked

    Returns:
        output: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# ======================================================================
# Multi-Head Attention
# ======================================================================

class MultiHeadAttention(nn.Module):
    """
