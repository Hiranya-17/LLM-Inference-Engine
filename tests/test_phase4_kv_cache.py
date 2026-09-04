"""
Phase 4 verification: proves generate_with_cache produces IDENTICAL output
to generate_naive (your own phase 3 code - no external reference needed
here, per the original spec), and measures the speedup.

Run with:  python3 -m pytest tests/test_phase4_kv_cache.py -v -s
"""
import os
import time

import pytest

from src.safetensors_reader import read_header, load_tensor
from src.bpe_tokenizer import load_byte_to_unicode, load_tokenizer_data, encode as bpe_encode, decode as bpe_decode
from src.forward_pass import CONFIG
from src.kv_cache import generate_naive, generate_with_cache
from bench.benchmark import record

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct")
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
]
NUM_NEW_TOKENS = 12


@pytest.fixture(scope="module")
def weights():
    tensors, data_start = read_header(SAFETENSORS_PATH)
    return {name: load_tensor(SAFETENSORS_PATH, info, data_start) for name, info in tensors.items()}


@pytest.fixture(scope="module")
def tokenizer_data():
    byte_encoder = load_byte_to_unicode()
    vocab, merge_ranks, pattern = load_tokenizer_data(TOKENIZER_PATH)
    return vocab, merge_ranks, byte_encoder, pattern


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_cached_output_identical_to_naive(weights, tokenizer_data, prompt):
    vocab, merge_ranks, byte_encoder, pattern = tokenizer_data
    input_ids = bpe_encode(prompt, vocab, merge_ranks, byte_encoder, pattern)

    naive_ids = generate_naive(input_ids, weights, CONFIG, NUM_NEW_TOKENS)
    cached_ids = generate_with_cache(input_ids, weights, CONFIG, NUM_NEW_TOKENS)

    byte_decoder = {v: k for k, v in byte_encoder.items()}
    print(f"\nprompt: {prompt!r}")
    print(f"  naive : {naive_ids} -> {bpe_decode(naive_ids, vocab, byte_decoder)!r}")
    print(f"  cached: {cached_ids} -> {bpe_decode(cached_ids, vocab, byte_decoder)!r}")

    assert cached_ids == naive_ids


def test_kv_cache_speedup(weights, tokenizer_data):
    vocab, merge_ranks, byte_encoder, pattern = tokenizer_data
    prompt = "The quick brown fox jumps over the lazy dog and runs into the forest"
    input_ids = bpe_encode(prompt, vocab, merge_ranks, byte_encoder, pattern)
    n = 20

    t0 = time.perf_counter()
    generate_naive(input_ids, weights, CONFIG, n)
    naive_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    generate_with_cache(input_ids, weights, CONFIG, n)
    cached_elapsed = time.perf_counter() - t0

    speedup = naive_elapsed / cached_elapsed
    print(f"\nnaive:  {naive_elapsed:.3f}s ({n/naive_elapsed:.2f} tok/s)")
    print(f"cached: {cached_elapsed:.3f}s ({n/cached_elapsed:.2f} tok/s)")
    print(f"speedup: {speedup:.2f}x")

    record(phase="phase4_generate_naive", tokens=n, elapsed_s=naive_elapsed)
    record(phase="phase4_generate_cached", tokens=n, elapsed_s=cached_elapsed, extra={"speedup_vs_naive": round(speedup, 2)})

    assert cached_elapsed < naive_elapsed, "KV cache should be faster than recomputing everything"
