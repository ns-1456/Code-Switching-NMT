#!/usr/bin/env python3
"""
Phase 5a: Inference — Greedy Decoding and Beam Search for English → Hinglish NMT.

Provides:
  - translate_greedy: simple argmax decoding (fast, for debugging)
  - translate_beam: beam search decoding (better quality, for demo/eval)
  - load_model_for_inference: loads checkpoint + tokenizer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from src.model import (
    Seq2SeqTransformer,
    build_model,
    create_padding_mask,
    create_tgt_mask,
    load_config,
)
from src.tokenizer import load_tokenizer


# ======================================================================
# Model Loading
# ======================================================================

def load_model_for_inference(
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "configs/config.yaml",
    device: torch.device | None = None,
) -> tuple[Seq2SeqTransformer, object, dict]:
    """
    Load a trained model from checkpoint.

    Returns:
        (model, tokenizer, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    tokenizer = load_tokenizer(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(config, vocab_size=tokenizer.vocab_size)

    # Filter out positional encoding buffers from checkpoint — they may
    # have been saved with a smaller size (e.g. 64) than the current model
    # uses (512). The PE buffer is deterministic (sinusoidal), so the fresh
    # one in the new model is identical for overlapping positions.
    saved_state = checkpoint["model_state_dict"]
    pe_keys = [k for k in saved_state if ".pe" in k]
    for k in pe_keys:
        if saved_state[k].shape != model.state_dict()[k].shape:
            del saved_state[k]

    model.load_state_dict(saved_state, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[inference] Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model, tokenizer, config


# ======================================================================
# Greedy Decoding
# ======================================================================

def translate_greedy(
    model: Seq2SeqTransformer,
    tokenizer,
    sentence: str,
    device: torch.device,
    max_len: int = 64,
    sos_id: int | None = None,
    eos_id: int | None = None,
    pad_idx: int | None = None,
) -> tuple[str, list[torch.Tensor]]:
    """
    Greedy decoding: pick argmax at each time step.

    Args:
        model: trained Seq2SeqTransformer
        tokenizer: trained tokenizer
        sentence: English input sentence
        device: torch device
        max_len: maximum output length
        sos_id, eos_id, pad_idx: special token IDs (auto-detected if None)

    Returns:
        (translated_text, cross_attention_weights)
    """
    if sos_id is None:
        sos_id = tokenizer.bos_token_id
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    if pad_idx is None:
        pad_idx = tokenizer.pad_token_id

    model.eval()

    # Encode source
    src_ids = tokenizer.encode(sentence, add_special_tokens=False)
    src_ids = [sos_id] + src_ids + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = create_padding_mask(src, pad_idx)

    enc_output = model.encode(src, src_mask)

    # Decode step by step
    tgt_ids = [sos_id]
    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        tgt_mask = create_tgt_mask(tgt, pad_idx)

        dec_output = model.decode(tgt, enc_output, tgt_mask, src_mask)
        logits = model.generator(dec_output[:, -1, :])
        next_token = logits.argmax(dim=-1).item()

        if next_token == eos_id:
            break
        tgt_ids.append(next_token)

    # Get attention weights for visualization
    attn_weights = model.get_cross_attention_weights()

    # Decode output tokens (skip <sos>)
    output_text = tokenizer.decode(tgt_ids[1:], skip_special_tokens=True)
    return output_text, attn_weights


# ======================================================================
# Beam Search Decoding
# ======================================================================

@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""
    tokens: list[int] = field(default_factory=list)
    log_prob: float = 0.0

    @property
    def score(self) -> float:
        """Length-normalized log probability."""
        # Add small epsilon to avoid division by zero for empty sequences
        length = max(1, len(self.tokens))
        return self.log_prob / length


def translate_beam(
    model: Seq2SeqTransformer,
    tokenizer,
    sentence: str,
    device: torch.device,
    beam_width: int = 5,
    max_len: int = 64,
    length_penalty: float = 1.0,
    sos_id: int | None = None,
    eos_id: int | None = None,
    pad_idx: int | None = None,
) -> tuple[str, list[torch.Tensor]]:
    """
    Beam search decoding: keep top-k hypotheses at each step.

    Args:
        model: trained Seq2SeqTransformer
        tokenizer: trained tokenizer
        sentence: English input sentence
        device: torch device
        beam_width: number of beams to keep
        max_len: maximum output length
        length_penalty: penalty factor for length normalization
        sos_id, eos_id, pad_idx: special token IDs

    Returns:
