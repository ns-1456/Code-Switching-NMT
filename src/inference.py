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

    # Initialize beams
    beams: list[BeamHypothesis] = [BeamHypothesis(tokens=[sos_id], log_prob=0.0)]
    completed: list[BeamHypothesis] = []

    for step in range(max_len):
        all_candidates: list[BeamHypothesis] = []

        for beam in beams:
            # If this beam already ended, skip
            if beam.tokens[-1] == eos_id:
                completed.append(beam)
                continue

            tgt = torch.tensor([beam.tokens], dtype=torch.long, device=device)
            tgt_mask = create_tgt_mask(tgt, pad_idx)

            dec_output = model.decode(tgt, enc_output, tgt_mask, src_mask)
            logits = model.generator(dec_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            # Get top-k candidates
            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                new_tokens = beam.tokens + [topk_ids[i].item()]
                new_log_prob = beam.log_prob + topk_log_probs[i].item()
                all_candidates.append(
                    BeamHypothesis(tokens=new_tokens, log_prob=new_log_prob)
                )

        if not all_candidates:
            break

        # Select top beam_width candidates (by length-normalized score)
        all_candidates.sort(
            key=lambda h: h.log_prob / (len(h.tokens) ** length_penalty),
            reverse=True,
        )
        beams = all_candidates[:beam_width]

        # Check if all beams are complete
        if all(b.tokens[-1] == eos_id for b in beams):
            completed.extend(beams)
            break

    # Add remaining beams to completed
    completed.extend(beams)

    # Pick best hypothesis
    if completed:
        best = max(
            completed,
            key=lambda h: h.log_prob / (max(1, len(h.tokens)) ** length_penalty),
        )
    else:
        best = beams[0] if beams else BeamHypothesis(tokens=[sos_id])

    # Run final forward pass to get attention weights
    tgt = torch.tensor([best.tokens], dtype=torch.long, device=device)
    tgt_mask = create_tgt_mask(tgt, pad_idx)
    _ = model.decode(tgt, enc_output, tgt_mask, src_mask)
    attn_weights = model.get_cross_attention_weights()

    # Decode tokens (skip <sos> and <eos>)
    token_ids = best.tokens[1:]  # skip <sos>
    if token_ids and token_ids[-1] == eos_id:
        token_ids = token_ids[:-1]
    output_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    return output_text, attn_weights


# ======================================================================
# Batched Greedy Decoding (FAST — for corpus evaluation)
# ======================================================================

def batch_translate_greedy(
    model: Seq2SeqTransformer,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    max_len: int = 64,
    batch_size: int = 128,
    sos_id: int | None = None,
    eos_id: int | None = None,
    pad_idx: int | None = None,
) -> list[str]:
    """
    Batched greedy decoding: translate many sentences in parallel on GPU.

    ~50-100x faster than one-by-one greedy for large test sets.

    Args:
        model: trained Seq2SeqTransformer
        tokenizer: trained tokenizer
        sentences: list of English input sentences
        device: torch device
        max_len: maximum output length
        batch_size: number of sentences to process at once
        sos_id, eos_id, pad_idx: special token IDs

    Returns:
        list of translated strings (same order as input)
    """
    if sos_id is None:
        sos_id = tokenizer.bos_token_id
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    if pad_idx is None:
        pad_idx = tokenizer.pad_token_id

    model.eval()
    all_outputs: list[str] = []

    for batch_start in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_start : batch_start + batch_size]
        bsz = len(batch_sents)

        # --- Encode all sources in this batch ---
        # Tokenize and add <sos>/<eos>
        src_id_lists = []
        for sent in batch_sents:
            ids = tokenizer.encode(sent, add_special_tokens=False)
            src_id_lists.append([sos_id] + ids + [eos_id])

        # Pad to max length in batch
        max_src = max(len(ids) for ids in src_id_lists)
        src_padded = torch.full((bsz, max_src), pad_idx, dtype=torch.long, device=device)
        for i, ids in enumerate(src_id_lists):
            src_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        src_mask = create_padding_mask(src_padded, pad_idx)  # (bsz, 1, 1, max_src)

        with torch.no_grad():
            enc_output = model.encode(src_padded, src_mask)  # (bsz, max_src, d_model)

        # --- Decode greedily in parallel ---
        # Start with <sos> for every sentence
        tgt_ids = torch.full((bsz, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = create_tgt_mask(tgt_ids, pad_idx)

            with torch.no_grad():
                dec_output = model.decode(tgt_ids, enc_output, tgt_mask, src_mask)
                logits = model.generator(dec_output[:, -1, :])  # (bsz, vocab)

            next_tokens = logits.argmax(dim=-1)  # (bsz,)

            # Force pad for already-finished sequences
            next_tokens[finished] = pad_idx

            # Check which sequences just hit <eos>
            finished = finished | (next_tokens == eos_id)

            tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # --- Decode each output ---
        for i in range(bsz):
            ids = tgt_ids[i, 1:].tolist()  # skip <sos>
            # Truncate at first <eos> or <pad>
            clean_ids = []
            for tid in ids:
                if tid == eos_id or tid == pad_idx:
                    break
                clean_ids.append(tid)
            all_outputs.append(tokenizer.decode(clean_ids, skip_special_tokens=True))

    return all_outputs


# ======================================================================
# Convenience wrapper
# ======================================================================

def translate(
    sentence: str,
    model: Seq2SeqTransformer | None = None,
    tokenizer=None,
    config: dict | None = None,
    device: torch.device | None = None,
    method: str = "beam",
    beam_width: int = 5,
) -> str:
    """
    High-level translate function. Loads model if not provided.

    Args:
        sentence: English input
        model: optional pre-loaded model
        tokenizer: optional pre-loaded tokenizer
        config: optional config dict
        method: 'greedy' or 'beam'
        beam_width: beam width for beam search

    Returns:
        Hinglish translation string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or tokenizer is None:
        model, tokenizer, config = load_model_for_inference(device=device)

    if method == "greedy":
        text, _ = translate_greedy(model, tokenizer, sentence, device)
    else:
        text, _ = translate_beam(model, tokenizer, sentence, device, beam_width=beam_width)

    return text


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    model, tokenizer, config = load_model_for_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sentences = [
