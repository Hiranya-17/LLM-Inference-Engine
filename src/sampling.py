"""
Temperature, top-k, and top-p sampling with a seedable RNG.

No `torch` or `transformers` here - those are reference-oracle-only, used in
tests/, never in this file.

--- Pipeline ---
sample(logits, rng, temperature, top_k, top_p):
  1. logits = apply_temperature(logits, temperature)
  2. if top_k is not None: logits = top_k_filter(logits, top_k)
  3. if top_p is not None: logits = top_p_filter(logits, top_p)
  4. probs = softmax(logits)
  5. return one token id, drawn from probs using rng

Use numpy's Generator objects for the RNG: rng = np.random.default_rng(seed).
Same seed -> same sequence of draws, every time.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Standard softmax: exponentiate then normalize. Subtract the max first
    for numerical stability (same trick as phase 3's attention) - it
    doesn't change the result, since it's a common factor that cancels out
    in the normalization.

    logits: (vocab_size,)
    Returns: (vocab_size,) - a valid probability distribution (sums to 1).
    """
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Divide logits by temperature. temperature < 1 sharpens the distribution
    (more confident / closer to greedy); temperature > 1 flattens it (more
    random). This is the entire function - one line.
    """
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the k highest logits; set every other logit to -inf (so they
    get probability 0 after softmax).

    Use exact top-k INDICES, not a value threshold - a threshold approach
    can accidentally keep more than k entries if there are ties.

    Hint: np.argsort(logits) sorts ascending and returns the INDICES that
    would sort the array (not the sorted values themselves). The last k
    indices of that result are the k largest entries' positions.

    logits: (vocab_size,)
    Returns: (vocab_size,) - a copy of logits, with all but the top k
             entries replaced by -inf.
    """
    sorted_indices = np.argsort(logits)
    top_k_indices = sorted_indices[-k:]
    result = np.full_like(logits, -np.inf)
    result[top_k_indices] = logits[top_k_indices]
    return result


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep the SMALLEST set of highest-probability tokens
    whose probabilities sum to at least p; set everything else to -inf.

    Steps:
      1. probs = softmax(logits)
      2. Sort indices by probability, DESCENDING:
         sorted_indices = np.argsort(-probs)   (negate to sort descending
         instead of ascending)
      3. sorted_probs = probs[sorted_indices]
      4. cumulative = np.cumsum(sorted_probs)  - running total as you walk
         down the sorted list.
      5. Find how many tokens to keep: the smallest count n such that
         cumulative[n-1] >= p. np.searchsorted(cumulative, p) gives you
         the first index where cumulative reaches p; add 1 for the count
         of tokens to keep (searchsorted returns an index, not a count).
      6. keep_indices = sorted_indices[:n]
      7. Build the result: copy logits, set every position EXCEPT
         keep_indices to -inf.

    logits: (vocab_size,)
    Returns: (vocab_size,)
    """
    probs = softmax(logits)
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    n = np.searchsorted(cumulative, p) + 1
    keep_indices = sorted_indices[:n]
    result = np.full_like(logits, -np.inf)
    result[keep_indices] = logits[keep_indices]
    return result


def sample(logits: np.ndarray, rng: np.random.Generator, temperature: float = 1.0, top_k: int = None, top_p: float = None) -> int:
    """
    Full sampling pipeline: apply_temperature, then top_k_filter if top_k
    is given, then top_p_filter if top_p is given, then softmax, then draw
    one token id from that distribution using rng.

    Hint: rng.choice(len(probs), p=probs) draws a single index according
    to the given probabilities.

    Returns: a single int, the sampled token id.
    """

    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = softmax(logits)
    return rng.choice(len(probs), p=probs)
