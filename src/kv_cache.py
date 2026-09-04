"""
KV-cached generation, built on top of phase 3's src/forward_pass.py.

The idea: K and V vectors for a given position never change once computed.
Instead of recomputing them for the whole sequence on every generation step
(what generate_naive does, and all phase 3 code did), cache them and only
compute K/V for the newest token(s) each step.

One masking rule handles both the initial "prefill" (the whole prompt at
once, cache starts empty) and later single-token "decode" steps: query
position i may attend to key position j whenever j <= i, using ABSOLUTE
positions in the overall sequence for both. Prefill naturally produces the
familiar lower-triangular mask; decode naturally produces "attend to
everything," since a freshly generated token can never be older than
anything already cached.
"""
from __future__ import annotations

import numpy as np

from src.forward_pass import embed, rms_norm, mlp, apply_rope, forward


def attention_with_cache(x: np.ndarray, weights: dict, layer_idx: int, position_start: int, config: dict, cache: list) -> np.ndarray:
    """
    Like phase 3's attention(), but reads/extends a persistent cache instead
    of recomputing K/V for positions it's already seen.

    x: (chunk_len, hidden_size) - the WHOLE prompt on the first (prefill)
       call, or just ONE new token's hidden state on every call after that.
    position_start: the absolute sequence position of x[0]. 0 for a fresh
       prefill; on later calls, wherever the sequence has grown to so far.
    cache: a list with one entry per layer. cache[layer_idx] is either None
       (nothing cached yet for this layer) or a dict {"k": ..., "v": ...}
       holding every position's key/value computed so far, shaped
       (total_len_so_far, num_kv_heads, head_dim).

    Steps:
      1. positions_for_x = np.arange(position_start, position_start + chunk_len)
         - the ABSOLUTE positions this call's x occupies.
      2. Project x through q/k/v and reshape into heads, same as phase 3's
         attention (same weight names, same biases).
      3. Apply apply_rope to q and k using positions_for_x (not starting
         from 0 - RoPE needs each token's true position in the sequence).
      4. If cache[layer_idx] is not None, concatenate its "k" and "v" with
         this call's new k and v (along the sequence axis, axis=0) - the
         old ones first, then the new ones. If cache[layer_idx] is None,
         the "concatenated" k/v are just this call's k/v.
      5. Store the concatenated k, v back into cache[layer_idx] (mutate it
         so the NEXT call sees everything up through this call).
      6. GQA-repeat the (now possibly longer than x) k and v, same idea as
         phase 3 - each kv head repeated 7 times to match 14 query heads.
      7. Causal mask, built from ABSOLUTE positions:
         key_positions = np.arange(total_len_so_far)
         mask[i, j] = key_positions[j] > positions_for_x[i]   (True = hide)
         Note this mask is NOT necessarily square - q only covers the new
         chunk_len positions, but k/v cover the full total_len_so_far.
      8. Per-head scaled dot-product attention with that mask, same as
         phase 3 (scores = q_head @ k_head.T / sqrt(head_dim), mask, softmax,
         weighted sum with v_head).
      9. Merge heads back to (chunk_len, hidden_size), project through
         o_proj (no bias).

    Returns: (chunk_len, hidden_size) - output for just the NEW positions
             passed in as x, not the whole cached history.
    """
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    chunk_len = x.shape[0]
    prefix = f"model.layers.{layer_idx}.self_attn."

    positions_for_x = np.arange(position_start, position_start + chunk_len)   # NEW: absolute positions

    # --- identical to phase 3's attention() from here... ---
    q = (x @ weights[prefix + "q_proj.weight"].T + weights[prefix + "q_proj.bias"]).reshape(chunk_len, num_heads, head_dim)
    k = (x @ weights[prefix + "k_proj.weight"].T + weights[prefix + "k_proj.bias"]).reshape(chunk_len, num_kv_heads, head_dim)
    v = (x @ weights[prefix + "v_proj.weight"].T + weights[prefix + "v_proj.bias"]).reshape(chunk_len, num_kv_heads, head_dim)

    q = apply_rope(q, positions_for_x, config["rope_theta"])
    k = apply_rope(k, positions_for_x, config["rope_theta"])
    # --- ...to here (only difference: positions_for_x instead of a plain arange) ---

    # NEW: read + extend the cache
    if cache[layer_idx] is not None:
        k = np.concatenate([cache[layer_idx]["k"], k], axis=0)
        v = np.concatenate([cache[layer_idx]["v"], v], axis=0)
    cache[layer_idx] = {"k": k, "v": v}

    k = np.repeat(k, num_heads // num_kv_heads, axis=1)   # same as phase 3, just on the longer k/v now
    v = np.repeat(v, num_heads // num_kv_heads, axis=1)

    total_len = k.shape[0]
    key_positions = np.arange(total_len)
    causal_mask = key_positions[None, :] > positions_for_x[:, None]   # NEW: non-square mask

    head_outputs = []
    for h in range(num_heads):
        q_head, k_head, v_head = q[:, h, :], k[:, h, :], v[:, h, :]
        scores = q_head @ k_head.T / np.sqrt(head_dim)
        scores[causal_mask] = -np.inf
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        head_outputs.append(attn_weights @ v_head)

    merged = np.stack(head_outputs, axis=1).reshape(chunk_len, num_heads * head_dim)
    return merged @ weights[prefix + "o_proj.weight"].T


def decoder_layer_with_cache(x: np.ndarray, weights: dict, layer_idx: int, position_start: int, config: dict, cache: list) -> np.ndarray:
    """
    Same structure as phase 3's decoder_layer (residual + attention block,
    residual + MLP block), but calls attention_with_cache instead of
    attention. The MLP block is IDENTICAL to phase 3's - it works purely
    per-position with no cross-token information, so it never needed
    caching at all.
    """
    prefix = f"model.layers.{layer_idx}."
    residual = x
    x = attention_with_cache(rms_norm(x, weights[prefix + "input_layernorm.weight"], config["rms_norm_eps"]), weights, layer_idx, position_start, config, cache)
    x = residual + x

    residual = x
    x = mlp(rms_norm(x, weights[prefix + "post_attention_layernorm.weight"], config["rms_norm_eps"]), weights, layer_idx)
    x = residual + x

    return x


def generate_naive(input_ids: list, weights: dict, config: dict, num_new_tokens: int) -> list:
    """
    The slow baseline: greedy-decode num_new_tokens new tokens by calling
    phase 3's forward() on the WHOLE sequence again every step.

    Loop num_new_tokens times:
      1. logits = forward(ids, weights, config)
      2. next_id = argmax of logits[-1] (the last position's prediction)
      3. append next_id to ids

    Returns the full list of ids: original input_ids + num_new_tokens more.
    """
    ids = list(input_ids)
    for _ in range(num_new_tokens):
        logits = forward(ids, weights, config)
        next_id = int(np.argmax(logits[-1]))
        ids.append(next_id)
    return ids


def generate_with_cache(input_ids: list, weights: dict, config: dict, num_new_tokens: int) -> list:
    """
    Greedy-decode num_new_tokens new tokens using the KV cache.

    1. cache = [None] * config["num_hidden_layers"]
    2. Prefill: run the WHOLE input_ids through embed + all layers'
       decoder_layer_with_cache (position_start=0), then final rms_norm and
       the tied-embedding projection to get logits, same shape math as
       phase 3's forward(). Take argmax of the LAST position's logits as
       the first new token.
    3. Loop num_new_tokens - 1 more times: embed just the ONE most recently
       generated token (as a length-1 sequence), run it through all layers'
       decoder_layer_with_cache with position_start = current length of the
       sequence so far, then rms_norm + projection, argmax, append.

    Returns the full list of ids, same shape/meaning as generate_naive's
    output - and for a correct implementation, IDENTICAL to it.
    """
    cache = [None] * config["num_hidden_layers"]
    ids = list(input_ids)

    # prefill: whole prompt at once, position_start=0
    x = embed(ids, weights["model.embed_tokens.weight"])
    for layer_idx in range(config["num_hidden_layers"]):
        x = decoder_layer_with_cache(x, weights, layer_idx, 0, config, cache)
    x = rms_norm(x, weights["model.norm.weight"], config["rms_norm_eps"])
    logits = x @ weights["model.embed_tokens.weight"].T
    ids.append(int(np.argmax(logits[-1])))

    # decode: one new token at a time
    for _ in range(num_new_tokens - 1):
        x = embed([ids[-1]], weights["model.embed_tokens.weight"])
        for layer_idx in range(config["num_hidden_layers"]):
            x = decoder_layer_with_cache(x, weights, layer_idx, len(ids) - 1, config, cache)
        x = rms_norm(x, weights["model.norm.weight"], config["rms_norm_eps"])
        logits = x @ weights["model.embed_tokens.weight"].T
        ids.append(int(np.argmax(logits[-1])))

    return ids