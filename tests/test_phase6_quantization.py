"""
Phase 6 verification (per-tensor quantization).

Part 1 checks quantize_tensor/dequantize_tensor against known math (no
external reference needed - the affine quantization formula IS the
definition of correctness here): the maximum-magnitude element always maps
to exactly +-qmax, and every dequantized value is within half a scale step
of the original (the fundamental rounding-error bound of quantization).

Part 2 loads the real model, quantizes it, and measures perplexity on a
short piece of text at fp32 (baseline), INT8, and INT4 - logged to
bench/results.jsonl. Perplexity is expected to get WORSE (higher) as
precision drops; this is measured and reported, not something with a
single "correct" numeric target.

Run with:  python3 -m pytest tests/test_phase6_quantization.py -v -s
"""
import os

import numpy as np
import pytest

from src.safetensors_reader import read_header, load_tensor
from src.bpe_tokenizer import load_byte_to_unicode, load_tokenizer_data, encode as bpe_encode
from src.forward_pass import CONFIG
from src.quantization import quantize_tensor, dequantize_tensor, quantize_weights, compute_perplexity
from bench.benchmark import record

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct")
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

PERPLEXITY_TEXT = (
    "The history of computing spans centuries, from early mechanical "
    "calculators to today's powerful digital machines. Each generation of "
    "technology built upon the discoveries of the one before it, gradually "
    "transforming how humans store, process, and share information."
)


# --- Part 1: quantize_tensor / dequantize_tensor math, no model needed ---

@pytest.mark.parametrize("num_bits,qmax", [(8, 127), (4, 7)])
def test_max_magnitude_element_maps_to_qmax(num_bits, qmax):
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((16, 16)).astype(np.float32)
    tensor[3, 5] = 10.0   # force a known maximum-magnitude element
    quantized, scale = quantize_tensor(tensor, num_bits)

    assert quantized[3, 5] == qmax
    assert scale == pytest.approx(10.0 / qmax, rel=1e-6)


@pytest.mark.parametrize("num_bits", [8, 4])
def test_dequantized_error_bounded_by_half_scale(num_bits):
    rng = np.random.default_rng(1)
    tensor = rng.standard_normal((32, 32)).astype(np.float32) * 5
    quantized, scale = quantize_tensor(tensor, num_bits)
    dequantized = dequantize_tensor(quantized, scale)

    max_error = np.abs(tensor - dequantized).max()
    assert max_error <= scale / 2 + 1e-4


@pytest.mark.parametrize("num_bits,qmax", [(8, 127), (4, 7)])
def test_quantized_values_stay_in_range(num_bits, qmax):
    rng = np.random.default_rng(2)
    tensor = rng.standard_normal((20, 20)).astype(np.float32) * 100
    quantized, _ = quantize_tensor(tensor, num_bits)

    assert quantized.min() >= -qmax
    assert quantized.max() <= qmax


def test_quantize_weights_leaves_1d_tensors_unchanged():
    weights = {
        "some.norm.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "some.linear.weight": np.random.default_rng(3).standard_normal((8, 8)).astype(np.float32),
    }
    result = quantize_weights(weights, num_bits=8)

    np.testing.assert_array_equal(result["some.norm.weight"], weights["some.norm.weight"])
    assert result["some.linear.weight"].shape == (8, 8)
    assert not np.array_equal(result["some.linear.weight"], weights["some.linear.weight"])


# --- Part 2: real model, perplexity at each precision level ---

@pytest.fixture(scope="module")
def weights():
    tensors, data_start = read_header(SAFETENSORS_PATH)
    return {name: load_tensor(SAFETENSORS_PATH, info, data_start) for name, info in tensors.items()}


@pytest.fixture(scope="module")
def token_ids():
    byte_encoder = load_byte_to_unicode()
    vocab, merge_ranks, pattern = load_tokenizer_data(TOKENIZER_PATH)
    return bpe_encode(PERPLEXITY_TEXT, vocab, merge_ranks, byte_encoder, pattern)


def test_perplexity_across_precision_levels(weights, token_ids):
    fp32_ppl = compute_perplexity(token_ids, weights, CONFIG)

    int8_weights = quantize_weights(weights, num_bits=8)
    int8_ppl = compute_perplexity(token_ids, int8_weights, CONFIG)

    int4_weights = quantize_weights(weights, num_bits=4)
    int4_ppl = compute_perplexity(token_ids, int4_weights, CONFIG)

    print(f"\nperplexity on {len(token_ids)} tokens:")
    print(f"  fp32: {fp32_ppl:.3f}")
    print(f"  int8: {int8_ppl:.3f}")
    print(f"  int4: {int4_ppl:.3f}")

    record(phase="phase6_quantization", tokens=len(token_ids), elapsed_s=0.0, extra={
        "fp32_perplexity": round(fp32_ppl, 3),
        "int8_perplexity": round(int8_ppl, 3),
        "int4_perplexity": round(int4_ppl, 3),
        "granularity": "per_tensor",
    })

    assert np.isfinite(fp32_ppl) and np.isfinite(int8_ppl) and np.isfinite(int4_ppl)
