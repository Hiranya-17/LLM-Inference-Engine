"""
INT8 and INT4 quantization - both per-tensor (one scale for the whole
matrix) and per-row (one scale per output neuron) - with perplexity
measured at each precision level and granularity.

No `torch` or `transformers` here - those are reference-oracle-only, used in
tests/, never in this file.

--- The idea: affine quantization ---
A float32 weight matrix uses 4 bytes per number and can represent an
enormous range of values. An n-bit signed integer can only represent
2^n distinct values (e.g. INT8: -128..127, though we use the symmetric
range -127..127 here; INT4: -7..7). To approximate a float tensor with
integers, we pick ONE scale factor per tensor:

    qmax = 2^(num_bits - 1) - 1          (127 for INT8, 7 for INT4)
    scale = max(abs(tensor)) / qmax      (float32, one number per tensor)

    to quantize:   int_value = round(tensor / scale), clipped to [-qmax, qmax]
    to dequantize: approx_tensor = int_value * scale

Every value in the tensor now shares the same scale, so quantizing loses
precision (many nearby float values collapse to the same integer) but
keeps the tensor roughly the same shape and range.

--- Per-row: a smaller blast radius for outliers ---
A single scale per tensor means one unusually large weight anywhere in the
matrix forces the scale up for EVERY value in it, crushing all the normal-
sized weights toward zero. Per-row quantization computes one scale per row
instead (each row = one output neuron's weights) - an outlier in one row
only hurts that row's own precision, leaving every other row's scale
tight and accurate.

--- "Fake" quantization ---
This project measures the ACCURACY impact of quantization, not the storage
savings - so quantize_weights immediately dequantizes back to float32
after quantizing. The forward pass still runs in float32, but on values
that have been rounded through an 8-bit or 4-bit bottleneck. This
"quantize-then-dequantize" technique is standard for measuring quantization
error (it's the core of quantization-aware training, for instance) even
though a real deployment would keep the integers packed in memory and use
specialized integer matmul kernels - that part is out of scope here.

--- Perplexity ---
Perplexity measures how "surprised" the model is by real text: at each
position, compute the model's predicted probability of the ACTUAL next
token, and average the surprise (negative log probability) across the
whole text. exp() of that average is the perplexity. Lower is better;
a perfect model that always assigns probability 1 to the true next token
has perplexity 1. Quantizing the weights should make the model slightly
worse at predicting real text, so perplexity should go up as num_bits
goes down.
"""
from __future__ import annotations

import numpy as np

from src.forward_pass import forward
from src.sampling import softmax


def quantize_tensor(tensor: np.ndarray, num_bits: int) -> tuple[np.ndarray, float]:
    """
    Affine (symmetric, per-tensor) quantization.

    qmax = 2^(num_bits - 1) - 1
    scale = max(abs(tensor)) / qmax
    quantized = round(tensor / scale), clipped to [-qmax, qmax], as int8

    tensor: any shape, float32
    num_bits: 8 or 4
    Returns: (quantized, scale) - quantized is the same shape as tensor,
             dtype int8 (used for both INT8 and INT4 - INT4 just uses a
             smaller slice of int8's range); scale is a single float.
    """
    qmax = 2 ** (num_bits - 1) - 1
    scale = np.abs(tensor).max() / qmax
    quantized = np.clip(np.round(tensor / scale), -qmax, qmax).astype(np.int8)
    return quantized, scale


def dequantize_tensor(quantized: np.ndarray, scale: float) -> np.ndarray:
    """
    Reconstruct an approximate float32 tensor from quantized ints + scale.

    approx = quantized (as float32) * scale

    Returns: float32 array, same shape as quantized.
    """
    return quantized.astype(np.float32) * scale


def quantize_weights(weights: dict, num_bits: int) -> dict:
    """
    Apply "fake" quantization (quantize then immediately dequantize) to
    every 2D weight matrix in the model. Leave 1D tensors (RMSNorm weights)
    untouched - they're small vectors, not worth quantizing, and this also
    conveniently separates "the big matrices we care about" from
    "everything else" using just tensor.ndim.

    weights: dict[str, np.ndarray] - the full model weights dict, same
             format as read by src/safetensors_reader.py.
    num_bits: 8 or 4

    Returns: a NEW dict[str, np.ndarray], same keys as weights. 2D tensors
             are replaced by their quantize->dequantize round trip; every
             other tensor is passed through unchanged.
    """
    result = {}
    for name, tensor in weights.items():
        if tensor.ndim == 2:
            quantized, scale = quantize_tensor(tensor, num_bits)
            result[name] = dequantize_tensor(quantized, scale)
        else:
            result[name] = tensor
    return result


def quantize_tensor_per_row(tensor: np.ndarray, num_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Like quantize_tensor, but with one scale PER ROW instead of one scale
    for the whole tensor. Each weight matrix here is shaped
    (out_features, in_features) - one row per output neuron - so a per-row
    scale means one outlier weight only stretches its own row's scale,
    instead of dragging the scale (and therefore the precision) of every
    other row in the tensor down with it.

    qmax = 2^(num_bits - 1) - 1
    row_max = max(abs(tensor)) along each row (axis=1), keeping the
              dimension so it still broadcasts against the full tensor:
              np.abs(tensor).max(axis=1, keepdims=True)
    scale = row_max / qmax                      (shape: (num_rows, 1))
    quantized = round(tensor / scale), clipped to [-qmax, qmax], as int8

    tensor: 2D, shape (num_rows, num_cols)
    num_bits: 8 or 4
    Returns: (quantized, scale) - quantized is the same shape as tensor,
             dtype int8; scale has shape (num_rows, 1), one value per row.
    """
    qmax = 2 ** (num_bits - 1) - 1
    row_max = np.abs(tensor).max(axis=1, keepdims=True)
    scale = row_max / qmax
    quantized = np.clip(np.round(tensor / scale), -qmax, qmax).astype(np.int8)
    return quantized, scale


def dequantize_tensor_per_row(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Same idea as dequantize_tensor, but scale is now (num_rows, 1) instead
    of a single float - it broadcasts against each row of quantized
    automatically.

    Returns: float32 array, same shape as quantized.
    """
    return quantized.astype(np.float32) * scale


def quantize_weights_per_row(weights: dict, num_bits: int) -> dict:
    """
    Same as quantize_weights, but using quantize_tensor_per_row /
    dequantize_tensor_per_row instead of the per-tensor versions.

    weights: dict[str, np.ndarray]
    num_bits: 8 or 4
    Returns: a NEW dict[str, np.ndarray], same keys as weights.
    """
    result = {}
    for name, tensor in weights.items():
        if tensor.ndim == 2:
            quantized, scale = quantize_tensor_per_row(tensor, num_bits)
            result[name] = dequantize_tensor_per_row(quantized, scale)
        else:
            result[name] = tensor
    return result


def compute_perplexity(ids: list, weights: dict, config: dict) -> float:
    """
    Perplexity of `weights` on the token sequence `ids`.

    Steps:
      1. logits = forward(ids, weights, config)   # (seq_len, vocab_size)
      2. For each position i from 0 to len(ids) - 2:
           probs = softmax(logits[i])              # model's prediction AT position i
           true_next_id = ids[i + 1]                # what actually came next
           accumulate -log(probs[true_next_id])     # "surprise" at this step
      3. average_surprise = total surprise / (len(ids) - 1)
      4. return exp(average_surprise)

    ids: list[int], length >= 2
    Returns: a single float >= 1.0 (in principle; can be large if the model
             predicts real text poorly).
    """
    logits = forward(ids, weights, config)
    total_surprise = 0.0
    for i in range(len(ids) - 1):
        probs = softmax(logits[i])
        total_surprise += -np.log(probs[ids[i + 1]])
    average_surprise = total_surprise / (len(ids) - 1)
    return float(np.exp(average_surprise))
