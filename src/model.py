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
    Multi-Head Attention with separate Q, K, V projections.

    Splits d_model into num_heads parallel attention heads, each with
    dimension d_k = d_model // num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Store attention weights for visualization
        self.attn_weights: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  (batch, 1, seq_q, seq_k) or broadcastable

        Returns:
            (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_weights.detach()

        # Concatenate heads and project
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.W_o(attn_output)


# ======================================================================
# Position-wise Feed-Forward Network
# ======================================================================

class FeedForward(nn.Module):
    """
    Two-layer FFN: Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ======================================================================
# Encoder Layer (Pre-Norm)
# ======================================================================

class EncoderLayer(nn.Module):
    """
    Single encoder layer: Self-Attention + FFN, with pre-norm and residuals.

    Pre-norm (LayerNorm before sublayer) is more stable for training
    compared to post-norm (original Vaswani).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len) padding mask
        """
        # Pre-norm self-attention
        _src = self.norm1(src)
        _src = self.self_attn(_src, _src, _src, mask=src_mask)
        src = src + self.dropout1(_src)

        # Pre-norm feed-forward
        _src = self.norm2(src)
        _src = self.ffn(_src)
        src = src + self.dropout2(_src)

        return src


# ======================================================================
# Decoder Layer (Pre-Norm)
# ======================================================================

class DecoderLayer(nn.Module):
    """
    Single decoder layer: Masked Self-Attention + Cross-Attention + FFN,
    with pre-norm and residuals.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch, tgt_len, d_model)
            enc_output: (batch, src_len, d_model)
            tgt_mask: (batch, 1, tgt_len, tgt_len) causal + padding mask
            src_mask: (batch, 1, 1, src_len) padding mask for encoder output
        """
        # Pre-norm masked self-attention
        _tgt = self.norm1(tgt)
        _tgt = self.self_attn(_tgt, _tgt, _tgt, mask=tgt_mask)
        tgt = tgt + self.dropout1(_tgt)

        # Pre-norm cross-attention (query from decoder, key/value from encoder)
        _tgt = self.norm2(tgt)
        _tgt = self.cross_attn(_tgt, enc_output, enc_output, mask=src_mask)
        tgt = tgt + self.dropout2(_tgt)

        # Pre-norm feed-forward
        _tgt = self.norm3(tgt)
        _tgt = self.ffn(_tgt)
        tgt = tgt + self.dropout3(_tgt)

        return tgt


# ======================================================================
# Encoder Stack
# ======================================================================

class Encoder(nn.Module):
    """Stack of N encoder layers with shared token embedding + positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Use a generous PE buffer (512) — it's a non-trainable buffer, costs nothing
        self.pos_encoding = PositionalEncoding(d_model, max(max_len * 4, 512), dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # Final norm (pre-norm architecture)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) token IDs
            src_mask: (batch, 1, 1, src_len)
        Returns:
            (batch, src_len, d_model)
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ======================================================================
# Decoder Stack
# ======================================================================

class Decoder(nn.Module):
    """Stack of N decoder layers with shared token embedding + positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Use a generous PE buffer (512) — it's a non-trainable buffer, costs nothing
        self.pos_encoding = PositionalEncoding(d_model, max(max_len * 4, 512), dropout)
