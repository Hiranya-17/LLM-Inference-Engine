"""
Phase 5 verification.

The deterministic parts (temperature scaling, top-k/top-p filtering) are
checked against transformers' LogitsWarper classes, used strictly as a
reference oracle - never imported by src/.

The actual random sampling can't be checked against a reference the same
way (different RNGs draw different random numbers even from identical
probabilities) - instead it's checked statistically (many draws should
approximate the known probabilities) and for reproducibility (same seed
-> same sequence).

Run with:  python3 -m pytest tests/test_phase5_sampling.py -v
"""
import numpy as np
import pytest
import torch
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from src.sampling import softmax, apply_temperature, top_k_filter, top_p_filter, sample

VOCAB_SIZE = 1000


def random_logits(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(VOCAB_SIZE).astype(np.float32) * 3


def assert_logits_match(yours, ref, atol=1e-3):
    """np.testing.assert_allclose chokes on comparing -inf to -inf, so check
    the -inf positions and the finite positions separately."""
    yours_inf = np.isneginf(yours)
    ref_inf = np.isneginf(ref)
    np.testing.assert_array_equal(yours_inf, ref_inf, err_msg="mismatch in which entries are -inf")
    np.testing.assert_allclose(yours[~yours_inf], ref[~ref_inf], atol=atol)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5, 2.0])
def test_apply_temperature_matches_reference(temperature):
    logits = random_logits(seed=0)
    yours = apply_temperature(logits, temperature)
    ref = TemperatureLogitsWarper(temperature)(None, torch.tensor(logits[None])).numpy()[0]
    np.testing.assert_allclose(yours, ref, atol=1e-3)


@pytest.mark.parametrize("k", [1, 5, 50, 200])
def test_top_k_filter_matches_reference(k):
    logits = random_logits(seed=1)
    yours = top_k_filter(logits, k)
    ref = TopKLogitsWarper(k)(None, torch.tensor(logits[None])).numpy()[0]
    assert_logits_match(yours, ref)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_top_p_filter_matches_reference(p):
    logits = random_logits(seed=2)
    yours = top_p_filter(logits, p)
    ref = TopPLogitsWarper(p)(None, torch.tensor(logits[None])).numpy()[0]
    assert_logits_match(yours, ref)


def test_softmax_sums_to_one_and_matches_torch():
    logits = random_logits(seed=3)
    probs = softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-6
    ref = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    np.testing.assert_allclose(probs, ref, atol=1e-5)


def test_sampling_matches_distribution_statistically():
    """Small, known distribution - many draws should approximate it."""
    logits = np.array([3.0, 1.0, 0.0, -1.0, -3.0])
    expected_probs = softmax(logits)

    rng = np.random.default_rng(42)
    n_draws = 100_000
    counts = np.zeros(len(logits))
    for _ in range(n_draws):
        counts[sample(logits, rng, temperature=1.0)] += 1
    empirical_probs = counts / n_draws

    np.testing.assert_allclose(empirical_probs, expected_probs, atol=0.01)


def test_seedable_rng_is_reproducible():
    logits = random_logits(seed=4)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    draws_a = [sample(logits, rng_a, temperature=1.0, top_k=10) for _ in range(50)]
    draws_b = [sample(logits, rng_b, temperature=1.0, top_k=10) for _ in range(50)]
    assert draws_a == draws_b


def test_top_k_1_is_equivalent_to_greedy():
    logits = random_logits(seed=5)
    expected = int(np.argmax(logits))
    rng = np.random.default_rng(0)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        assert sample(logits, rng, temperature=1.0, top_k=1) == expected
