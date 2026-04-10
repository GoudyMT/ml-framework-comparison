"""
Transformer utilities for Models.
Mask creation, scheduling, and decoding helpers used across
PyTorch Transformer pipelines.
"""

import numpy as np
import torch
from utils.attention_utils import compute_bleu

def create_pad_mask(seq, pad_idx):
    """
    Create padding mask: True where tokens are PAD.

    Prevents attention from attending to padding positions.
    Shape designed for broadcasting with attention scores
    of shape (batch, n_heads, query_len, key_len).

    Args:
        seq: Token indices (batch, seq_len).
        pad_idx: Index of the PAD token.

    Returns:
        Boolean mask (batch, 1, 1, seq_len). True = masked position.
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len, device):
    """
    Create causal (look-ahead) mask for decoder self-attention.

    Prevents position i from attending to positions j > i.
    This enforces autoregressive behavior — each token can only
    depend on previous tokens, not future ones.

    Args:
        seq_len: Length of the target sequence.
        device: Torch device.

    Returns:
        Boolean mask (1, 1, seq_len, seq_len). True = masked (future).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def greedy_decode(model, src, max_len, bos_idx, eos_idx, pad_idx, device):
    """
    Autoregressively generate a translation using greedy decoding.

    1. Encode the source sequence once
    2. Start with BOS token
    3. At each step, feed all generated tokens through decoder
    4. Take argmax of last position's logits as next token
    5. Stop at EOS or max_len

    Args:
        model: Trained Transformer with encode() and decode() methods.
        src: Source tokens (1, src_len) — single sentence.
        max_len: Maximum generation length.
        bos_idx: Beginning-of-sequence token index.
        eos_idx: End-of-sequence token index.
        pad_idx: Padding token index.
        device: Torch device.

    Returns:
        List of generated token IDs (excluding BOS).
    """
    model.eval()
    src = src.to(device)
    src_mask = create_pad_mask(src, pad_idx).to(device)

    with torch.no_grad():
        encoder_out = model.encode(src, src_mask)

    generated = [bos_idx]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_pad_mask = create_pad_mask(tgt_tensor, pad_idx).to(device)
        tgt_causal = create_causal_mask(tgt_tensor.size(1), device)
        tgt_mask = tgt_pad_mask | tgt_causal

        with torch.no_grad():
            logits = model.decode(tgt_tensor, encoder_out, src_mask, tgt_mask)

        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)

        if next_token == eos_idx:
            break

    return generated[1:]

def compute_bleu_greedy(model, src_data, tgt_data, sp_model, max_len,
                        bos_idx, eos_idx, pad_idx, device, max_samples=None):
    """
    Translate source sentences with greedy decode, compute corpus BLEU.

    BPE tokens are decoded back to words via sp_model.decode() before
    BLEU computation — ensures fair comparison with Attention #15's
    word-level BLEU.

    Args:
        model: Trained Transformer with encode()/decode() methods.
        src_data: Source tensor (n_samples, src_len).
        tgt_data: Target tensor (n_samples, tgt_len).
        sp_model: SentencePiece model for BPE decoding.
        max_len: Max generation length.
        bos_idx, eos_idx, pad_idx: Special token indices.
        device: Torch device.
        max_samples: Limit evaluation to first N samples (None = all).

    Returns:
        bleu_score: Corpus BLEU (float).
        translations: List of (source_text, reference_text, hypothesis_text).
    """
    model.eval()
    n = src_data.size(0) if max_samples is None else min(max_samples, src_data.size(0))

    references = []
    hypotheses = []
    translations = []

    for i in range(n):
        src_sent = src_data[i:i+1]

        pred_ids = greedy_decode(model, src_sent, max_len, bos_idx, eos_idx, pad_idx, device)

        # Decode BPE → words (filter special tokens before decoding)
        pred_clean = [t for t in pred_ids if t >= 4 and t != eos_idx]
        ref_ids = tgt_data[i].cpu().tolist()
        ref_clean = [t for t in ref_ids if t >= 4 and t != eos_idx and t != pad_idx]

        pred_text = sp_model.decode(pred_clean)
        ref_text = sp_model.decode(ref_clean)
        src_text = sp_model.decode([t for t in src_data[i].cpu().tolist() if t >= 4 and t != pad_idx])

        references.append([ref_text.split()])
        hypotheses.append(pred_text.split())
        translations.append((src_text, ref_text, pred_text))

    bleu = compute_bleu(references, hypotheses) 
    return bleu, translations

def beam_search_decode(model, src, beam_size, max_len, bos_idx, eos_idx,
                       pad_idx, device, length_penalty=0.6):
    """
    Beam search decoding for Transformer translation.

    Maintains k hypotheses at each step. At step t, expands every
    active hypothesis by considering its top-k next tokens, then
    keeps the global top-k by score (sum of log-probs, normalized
    by length).

    Length normalization: shorter hypotheses naturally accumulate
    higher probability (fewer tokens to multiply). Length penalty
    prevents the search from favoring shorter translations:
        normalized_score = log_prob / length^alpha
    where alpha=0.6 is Google NMT's recommended value.

    Args:
        model: Trained Transformer with encode()/decode() methods.
        src: Source tokens (1, src_len) -- single sentence.
        beam_size: Number of hypotheses to maintain (k).
        max_len: Maximum generation length.
        bos_idx: Beginning-of-sequence token index.
        eos_idx: End-of-sequence token index.
        pad_idx: Padding token index.
        device: Torch device.
        length_penalty: Alpha for length normalization (0.6 is standard).

    Returns:
        List of best hypothesis token IDs (excluding BOS).
    """
    model.eval()
    src = src.to(device)
    src_mask = create_pad_mask(src, pad_idx).to(device)

    with torch.no_grad():
        encoder_out = model.encode(src, src_mask)

    # Each beam: (token_ids, cumulative_log_prob, finished)
    beams = [([bos_idx], 0.0, False)]
    finished_beams = []

    for step in range(max_len):
        # If all beams finished, stop
        if all(b[2] for b in beams):
            break

        candidates = []

        for tokens, log_prob, finished in beams:
            if finished:
                candidates.append((tokens, log_prob, True))
                continue

            tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_pad_mask = create_pad_mask(tgt_tensor, pad_idx).to(device)
            tgt_causal = create_causal_mask(tgt_tensor.size(1), device)
            tgt_mask = tgt_pad_mask | tgt_causal

            with torch.no_grad():
                logits = model.decode(tgt_tensor, encoder_out, src_mask, tgt_mask)

            # Get log-probs for the last position
            last_logits = logits[0, -1, :]
            log_probs = torch.log_softmax(last_logits, dim=-1)

            # Top k next tokens
            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for k in range(beam_size):
                next_token = topk_indices[k].item()
                next_log_prob = log_prob + topk_log_probs[k].item()
                next_tokens = tokens + [next_token]
                next_finished = (next_token == eos_idx)
                candidates.append((next_tokens, next_log_prob, next_finished))

        # Length-normalized score for ranking
        def normalized_score(beam):
            tokens, lp, _ = beam
            length = max(len(tokens) - 1, 1)  # exclude BOS
            return lp / (length ** length_penalty)

        # Keep top beam_size candidates
        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:beam_size]

        # Move finished beams to finished list
        still_active = []
        for b in beams:
            if b[2]:
                finished_beams.append(b)
            else:
                still_active.append(b)
        beams = still_active

        if len(beams) == 0:
            break

    # Combine remaining active + finished, pick best by normalized score
    all_beams = finished_beams + beams
    if len(all_beams) == 0:
        return []
    best = max(all_beams, key=lambda b: b[1] / max(len(b[0]) - 1, 1) ** length_penalty)

    return best[0][1:]  # exclude BOS