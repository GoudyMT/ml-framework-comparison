"""
Attention-specific computation utilities.
Reusable for Attention (#15), Transformers (#16).

Functions:
    compute_bleu — Corpus-level BLEU score via nltk
    bleu_by_length — BLEU scores binned by source sentence length
"""

import numpy as np


def compute_bleu(references, hypotheses, max_n=4):
    """
    Compute corpus-level BLEU score using nltk.

    BLEU (Bilingual Evaluation Understudy) measures translation quality
    by comparing n-gram overlap between machine and reference translations.
    Corpus-level BLEU averages over all sentence pairs — more stable than
    sentence-level BLEU which is noisy on short sequences.

    Args:
        references: List of reference token lists, each wrapped in a list
                    (nltk expects [[ref_tokens]] per sentence to support
                    multiple references — we use one reference per pair).
        hypotheses: List of hypothesis (predicted) token lists.
        max_n: Maximum n-gram order (default 4 = standard BLEU-4).

    Returns:
        BLEU score as float in [0, 1]. Higher = better translation quality.
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # SmoothingFunction handles zero n-gram counts on short sentences
    # Without smoothing, a single missing 4-gram zeros out the entire score
    smoothing = SmoothingFunction().method1

    # Weights for n-gram precision (uniform = standard BLEU-4)
    weights = tuple(1.0 / max_n for _ in range(max_n))

    return corpus_bleu(references, hypotheses,
                       weights=weights,
                       smoothing_function=smoothing)


def bleu_by_length(references, hypotheses, src_lengths, n_buckets=5):
    """
    Compute BLEU score per source sentence length bucket.

    This reveals attention's key advantage: no-attention models degrade
    on longer sequences (bottleneck), while attention maintains quality.
    Binning by source length makes this visible.

    Args:
        references: List of [[ref_tokens]] (same format as compute_bleu).
        hypotheses: List of [hyp_tokens].
        src_lengths: List/array of source sentence token counts.
        n_buckets: Number of length bins (default 5).

    Returns:
        Dict with:
            'bucket_labels': List of str labels like '1-5'.
            'bleu_scores': List of float BLEU per bucket.
            'counts': List of int sentences per bucket.
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    smoothing = SmoothingFunction().method1
    src_lengths = np.array(src_lengths)

    # Create equal-frequency bins (each bucket has ~same number of sentences)
    percentiles = np.linspace(0, 100, n_buckets + 1)
    bin_edges = np.percentile(src_lengths, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicate edges

    bucket_labels = []
    bleu_scores = []
    counts = []

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (src_lengths >= low) & (src_lengths <= high)
        else:
            mask = (src_lengths >= low) & (src_lengths < high)

        bucket_refs = [references[j] for j in range(len(references)) if mask[j]]
        bucket_hyps = [hypotheses[j] for j in range(len(hypotheses)) if mask[j]]

        if len(bucket_refs) > 0:
            score = corpus_bleu(bucket_refs, bucket_hyps,
                                smoothing_function=smoothing)
        else:
            score = 0.0

        bucket_labels.append(f"{int(low)}-{int(high)}")
        bleu_scores.append(score)
        counts.append(int(mask.sum()))

    return {
        'bucket_labels': bucket_labels,
        'bleu_scores': bleu_scores,
        'counts': counts
    }
