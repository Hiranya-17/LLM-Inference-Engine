"""
Phase 1 verification: checks YOUR src/safetensors_reader.py against an
independent reference (the `safetensors` + `ml_dtypes` packages).

This is the only place in the whole project those packages are allowed to
appear - they are a verification oracle only, never part of the actual
implementation. src/ must never import `safetensors`, `ml_dtypes`, `torch`,
or `transformers`.

Run with:  python3 -m pytest tests/test_phase1_safetensors.py -v
"""
import os
import re

import numpy as np
import ml_dtypes  # noqa: F401  (side effect: registers bfloat16 with numpy)
import pytest
from safetensors import safe_open

from src.safetensors_reader import read_header, load_tensor

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct", "model.safetensors"
)


@pytest.fixture(scope="module")
def reference():
    with safe_open(MODEL_PATH, framework="numpy") as f:
        yield f


@pytest.fixture(scope="module")
def yours():
    tensors, data_start = read_header(MODEL_PATH)
    return tensors, data_start


def test_tensor_names_match_reference(yours, reference):
    your_names = set(yours[0].keys())
    ref_names = set(reference.keys())
    assert your_names == ref_names, (
        f"Missing from yours: {ref_names - your_names}\n"
        f"Extra in yours: {your_names - ref_names}"
    )


def test_shapes_and_dtypes_match_reference(yours, reference):
    tensors, _ = yours
    mismatches = []
    for name, info in tensors.items():
        ref_slice = reference.get_slice(name)
        ref_shape = tuple(ref_slice.get_shape())
        ref_dtype = ref_slice.get_dtype()
        if tuple(info.shape) != ref_shape or info.dtype != ref_dtype:
            mismatches.append((name, info.shape, info.dtype, ref_shape, ref_dtype))
    assert not mismatches, "\n".join(
        f"{n}: yours={s}/{d}  reference={rs}/{rd}" for n, s, d, rs, rd in mismatches
    )


def test_data_blob_fully_covered_no_overlap(yours):
    """Every byte after the header must belong to exactly one tensor."""
    tensors, data_start = yours
    spans = sorted(info.data_offsets for info in tensors.values())
    file_size = os.path.getsize(MODEL_PATH)
    expected_blob_size = file_size - data_start

    assert spans[0][0] == 0, f"data blob doesn't start at 0: {spans[0]}"
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 == s2, f"gap or overlap between offsets {e1} and {s2}"
    assert spans[-1][1] == expected_blob_size, (
        f"last tensor ends at {spans[-1][1]}, but blob is {expected_blob_size} bytes"
    )


@pytest.mark.parametrize("name", [
    "model.layers.0.input_layernorm.weight",   # small, 1-D
    "model.layers.0.self_attn.q_proj.weight",  # medium, 2-D
    "model.embed_tokens.weight",               # large, 151936 x 896
])
def test_bf16_upcast_is_bit_exact(yours, reference, name):
    """
    bf16 -> fp32 is a lossless bit operation (zero-fill the low 16 mantissa
    bits) - not an approximation. So this must match the reference EXACTLY,
    not within some tolerance.
    """
    tensors, data_start = yours
    your_array = load_tensor(MODEL_PATH, tensors[name], data_start)
    ref_array = reference.get_tensor(name).astype(np.float32)

    assert your_array.shape == ref_array.shape
    assert your_array.dtype == np.float32
    np.testing.assert_array_equal(your_array, ref_array)


def test_structural_sanity_against_config(yours):
    """Cross-check against config.json's stated architecture."""
    tensors, _ = yours

    layer_indices = set()
    for name in tensors:
        m = re.match(r"model\.layers\.(\d+)\.", name)
        if m:
            layer_indices.add(int(m.group(1)))
    assert layer_indices == set(range(24)), f"expected layers 0..23, got {sorted(layer_indices)}"

    assert tensors["model.embed_tokens.weight"].shape == (151936, 896)

    # tie_word_embeddings=true in config.json -> no separate output head
    assert "lm_head.weight" not in tensors
