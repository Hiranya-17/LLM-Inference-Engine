"""
A from-scratch naive float32 forward pass for Qwen2.5-0.5B-Instruct.

No `torch` or `transformers` here - those are reference-oracle-only, used in
tests/, never in this file. Everything below is NumPy.

--- Config (from config.json, phase 1) ---
hidden_size=896, num_attention_heads=14, num_key_value_heads=2, head_dim=64
(896/14=64), intermediate_size=4864, num_hidden_layers=24, rms_norm_eps=1e-6,
rope_theta=1000000.0, vocab_size=151936, tie_word_embeddings=true.

--- The pipeline ---
forward(input_ids):
  1. embed(input_ids) -> (seq_len, 896)
  2. for each of the 24 layers:
       residual = x
       x = attention(rms_norm(x, input_layernorm_weight)) ; x = residual + x
       residual = x
       x = mlp(rms_norm(x, post_attention_layernorm_weight)) ; x = residual + x
  3. x = rms_norm(x, model.norm.weight)
  4. logits = x @ model.embed_tokens.weight.T   (tied embeddings, no separate lm_head)
"""
from __future__ import annotations

import numpy as np

CONFIG = {
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "vocab_size": 151936,
}


def embed(input_ids: list, embed_weight: np.ndarray) -> np.ndarray:
    """
    Look up each token id's row in the embedding table.

    input_ids: list/array of int token ids, length seq_len.
    embed_weight: the (vocab_size, hidden_size) embedding matrix
                  (model.embed_tokens.weight).

    Returns: (seq_len, hidden_size) array - one row per input token.

    Hint: NumPy arrays can be indexed with a list of indices directly:
          arr[[2, 0, 5]] gives you rows 2, 0, and 5, in that order, as a
          new (3, ...) array. That's the entire function.
    """
    return embed_weight[input_ids]


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """
    RMSNorm: normalize each row of x by its root-mean-square magnitude,
    then scale by a learned per-channel weight. No mean-subtraction, no
    bias - simpler than LayerNorm.

    For each row (each position's hidden_size-length vector):
        rms = sqrt(mean(row ** 2) + eps)
        normalized_row = row / rms
        output_row = normalized_row * weight   (elementwise, weight is
                     also hidden_size-length - same weight for every row)

    x: (seq_len, hidden_size)
    weight: (hidden_size,)
    Returns: (seq_len, hidden_size)

    Hint: do this for ALL rows at once with NumPy, not a Python loop.
          x ** 2, then .mean(axis=-1, keepdims=True) averages each row
          down to one number while keeping it a (seq_len, 1) shape (so it
          still broadcasts against the (seq_len, hidden_size) x).
    """
    mean_sq = (x ** 2).mean(axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + eps)
    normalized = x / rms
    return normalized * weight

def apply_rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply rotary position embeddings to x (rotates each head's vector by
    an angle proportional to its position, instead of adding a position
    vector).

    x: (seq_len, num_heads, head_dim) - this is applied to q or k after
       they've been split into heads, before the attention dot-product.
    positions: (seq_len,) array of position indices, e.g. [0, 1, 2, ...]
    theta: CONFIG["rope_theta"]

    Algorithm (the "rotate half" convention Qwen/Llama use):
      1. Compute a frequency for each of the head_dim/2 dimension-pairs:
             freqs[i] = 1.0 / (theta ** (2*i / head_dim))   for i in 0..head_dim/2 - 1
      2. For each position p and each freq i, the rotation angle is
         angle[p, i] = p * freqs[i].
      3. Build cos and sin tables of shape (seq_len, head_dim/2) from
         those angles, then tile each to (seq_len, head_dim) by
         concatenating the table with itself along the last axis
         (so cos/sin repeat once - this matches how the two halves of x
         are rotated using the SAME per-pair angle).
      4. Split x's last dimension in half: x1 = x[..., :head_dim/2],
         x2 = x[..., head_dim/2:].
      5. "Rotate half" of x means building rotated_x = concat(-x2, x1)
         along the last axis (same shape as x, just the two halves
         swapped and the (originally second) half negated).
      6. Final result: x * cos + rotated_x * sin
         (cos/sin need a broadcastable shape - (seq_len, 1, head_dim) -
         to line up against x's (seq_len, num_heads, head_dim)).

    Returns: (seq_len, num_heads, head_dim), same shape as input x.
    """
    head_dim = x.shape[-1]
    i = np.arange(head_dim // 2)
    freqs = 1.0 / (theta ** (2 * i / head_dim))
    angles = positions[:, None] * freqs[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[:, None, :]
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[:, None, :]

    half = head_dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated_x = np.concatenate([-x2, x1], axis=-1)

    return x * cos + rotated_x * sin



def attention(x: np.ndarray, weights: dict, layer_idx: int, positions: np.ndarray, config: dict) -> np.ndarray:
    """
    Grouped-query self-attention for one layer.

    x: (seq_len, hidden_size) - already normalized (rms_norm applied by caller)
    weights: dict of all model tensors, keyed by their safetensors names
             (e.g. weights["model.layers.0.self_attn.q_proj.weight"])
    layer_idx: which of the 24 layers (0-23)
    positions: (seq_len,) position indices for RoPE
    config: CONFIG dict

    Steps:
      1. Project x through q_proj, k_proj, v_proj (each has a bias -
         don't forget to add it: x @ weight.T + bias).
         q: (seq_len, 896) -> reshape to (seq_len, 14, 64)
         k, v: (seq_len, 128) -> reshape to (seq_len, 2, 64)
      2. Apply apply_rope to q and k (not v).
      3. GQA expansion: each of the 2 kv heads must be repeated 7 times
         (14 query heads / 2 kv heads = 7) so k and v end up shaped
         (seq_len, 14, 64) too, matching q. Repeating means: kv head 0
         is reused as query-heads 0-6's key/value, kv head 1 for
         query-heads 7-13 (i.e. np.repeat along the heads axis with
         repeats=7).
      4. For each of the 14 heads independently: compute
         scores = q_head @ k_head.T / sqrt(64)   -> (seq_len, seq_len)
         Apply a CAUSAL mask: position i may only attend to positions
         <= i (set scores[i, j] = -inf for j > i, before softmax).
         Softmax over the last axis, then attn_head = softmax_scores @ v_head.
      5. Concatenate all 14 heads' outputs back into (seq_len, 896).
      6. Project through o_proj (no bias this time): result @ o_proj_weight.T

    Returns: (seq_len, hidden_size)
    """
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    seq_len = x.shape[0]
    prefix = f"model.layers.{layer_idx}.self_attn."

    q = (x @ weights[prefix + "q_proj.weight"].T + weights[prefix + "q_proj.bias"]).reshape(seq_len, num_heads, head_dim)
    k = (x @ weights[prefix + "k_proj.weight"].T + weights[prefix + "k_proj.bias"]).reshape(seq_len, num_kv_heads, head_dim)
    v = (x @ weights[prefix + "v_proj.weight"].T + weights[prefix + "v_proj.bias"]).reshape(seq_len, num_kv_heads, head_dim)

    q = apply_rope(q, positions, config["rope_theta"])
    k = apply_rope(k, positions, config["rope_theta"])

    k = np.repeat(k, num_heads // num_kv_heads, axis=1)
    v = np.repeat(v, num_heads // num_kv_heads, axis=1)

    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    head_outputs = []
    for h in range(num_heads):
        q_head, k_head, v_head = q[:, h, :], k[:, h, :], v[:, h, :]
        scores = q_head @ k_head.T / np.sqrt(head_dim)
        scores[causal_mask] = -np.inf
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        head_outputs.append(attn_weights @ v_head)

    merged = np.stack(head_outputs, axis=1).reshape(seq_len, num_heads * head_dim)
    return merged @ weights[prefix + "o_proj.weight"].T


def mlp(x: np.ndarray, weights: dict, layer_idx: int) -> np.ndarray:
    """
    SwiGLU MLP block: down_proj(silu(gate_proj(x)) * up_proj(x))

    silu(z) = z * sigmoid(z) = z / (1 + exp(-z))

    x: (seq_len, hidden_size), already normalized by caller
    weights: dict of all model tensors
    layer_idx: which of the 24 layers

    gate_proj, up_proj: (hidden_size -> intermediate_size), no bias
    down_proj: (intermediate_size -> hidden_size), no bias

    Returns: (seq_len, hidden_size)
    """
    prefix = f"model.layers.{layer_idx}.mlp."
    gate = x @ weights[prefix + "gate_proj.weight"].T
    up = x @ weights[prefix + "up_proj.weight"].T
    silu_gate = gate * (1 / (1 + np.exp(-gate)))
    return (silu_gate * up) @ weights[prefix + "down_proj.weight"].T


def decoder_layer(x: np.ndarray, weights: dict, layer_idx: int, positions: np.ndarray, config: dict) -> np.ndarray:
    """
    One full transformer layer: attention block then MLP block, each
    wrapped in its own residual connection.

        residual = x
        x = attention(rms_norm(x, input_layernorm_weight, eps), ...)
        x = residual + x

        residual = x
        x = mlp(rms_norm(x, post_attention_layernorm_weight, eps), ...)
        x = residual + x

    Returns: (seq_len, hidden_size)
    """
    prefix = f"model.layers.{layer_idx}."
    residual = x
    x = attention(rms_norm(x, weights[prefix + "input_layernorm.weight"], config["rms_norm_eps"]), weights, layer_idx, positions, config)
    x = residual + x

    residual = x
    x = mlp(rms_norm(x, weights[prefix + "post_attention_layernorm.weight"], config["rms_norm_eps"]), weights, layer_idx)
    x = residual + x

    return x


def forward(input_ids: list, weights: dict, config: dict = CONFIG) -> np.ndarray:
    """
    Full forward pass: token ids -> logits over the vocabulary.

      1. x = embed(input_ids, weights["model.embed_tokens.weight"])
      2. positions = np.arange(len(input_ids))
      3. Run x through all 24 decoder_layer calls in sequence (layer 0's
         output is layer 1's input, and so on).
      4. x = rms_norm(x, weights["model.norm.weight"], config["rms_norm_eps"])
      5. logits = x @ weights["model.embed_tokens.weight"].T
         (tied embeddings: reuse the same table transposed as the output
         head, since tie_word_embeddings=true means there's no separate
         lm_head.weight tensor)

    Returns: (seq_len, vocab_size)
    """
    x = embed(input_ids, weights["model.embed_tokens.weight"])
    positions = np.arange(len(input_ids))
    for layer_idx in range(config["num_hidden_layers"]):
        x = decoder_layer(x, weights, layer_idx, positions, config)
    x = rms_norm(x, weights["model.norm.weight"], config["rms_norm_eps"])
    return x @ weights["model.embed_tokens.weight"].T
