"""
Phase 3 verification: checks YOUR src/forward_pass.py against the reference
(transformers + torch, CPU, float32), used strictly as a verification oracle.
Never imported by src/.

Tokenization uses YOUR OWN phase 2 bpe_tokenizer, not transformers' tokenizer -
transformers is only used here to compute reference logits for the exact
token ids your own tokenizer produces.

Checks every layer's hidden state against the reference (not just final
logits) so a mismatch points at exactly which piece is wrong.

Run with:  python3 -m pytest tests/test_phase3_forward_pass.py -v
"""
import os

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from src.safetensors_reader import read_header, load_tensor
from src.bpe_tokenizer import load_byte_to_unicode, load_tokenizer_data, encode as bpe_encode
from src.forward_pass import embed, rms_norm, decoder_layer, forward, CONFIG

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct")
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "2 + 2 =",
]

ATOL = 1e-3


@pytest.fixture(scope="module")
def weights():
    tensors, data_start = read_header(SAFETENSORS_PATH)
    return {name: load_tensor(SAFETENSORS_PATH, info, data_start) for name, info in tensors.items()}


@pytest.fixture(scope="module")
def tokenized_prompts():
    byte_encoder = load_byte_to_unicode()
    vocab, merge_ranks, pattern = load_tokenizer_data(TOKENIZER_PATH)
    return [bpe_encode(p, vocab, merge_ranks, byte_encoder, pattern) for p in TEST_PROMPTS]


@pytest.fixture(scope="module")
def reference_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    return model


def run_reference(model, input_ids):
    with torch.no_grad():
        out = model(torch.tensor([input_ids]), output_hidden_states=True)
    return {
        "hidden_states": [h[0].numpy() for h in out.hidden_states],  # 25 arrays: (seq_len, 896)
        "logits": out.logits[0].numpy(),  # (seq_len, vocab_size)
    }


@pytest.mark.parametrize("prompt_idx", range(len(TEST_PROMPTS)))
def test_embeddings_match_reference(weights, tokenized_prompts, reference_model, prompt_idx):
    input_ids = tokenized_prompts[prompt_idx]
    ref = run_reference(reference_model, input_ids)
    your_emb = embed(input_ids, weights["model.embed_tokens.weight"])
    np.testing.assert_allclose(your_emb, ref["hidden_states"][0], atol=ATOL)


@pytest.mark.parametrize("prompt_idx", range(len(TEST_PROMPTS)))
def test_each_layer_matches_reference(weights, tokenized_prompts, reference_model, prompt_idx):
    """
    NOTE: HF's output_hidden_states has a quirk - hidden_states[N] (the last
    entry, N = num_hidden_layers) is the state AFTER model.norm is applied,
    not the raw output of the last layer like every other entry is. So the
    last layer is checked separately, post-norm, to match that semantics.

    Tolerance is looser than ATOL here on purpose: this per-layer check is
    an extra diagnostic beyond what was actually asked for (final logits
    within 1e-3) - tiny cross-library floating point differences in matmul
    ordering compound slightly with depth, so intermediate layers need a
    bit more room than the final output does.
    """
    input_ids = tokenized_prompts[prompt_idx]
    ref = run_reference(reference_model, input_ids)
    positions = np.arange(len(input_ids))
    num_layers = CONFIG["num_hidden_layers"]

    x = embed(input_ids, weights["model.embed_tokens.weight"])
    for layer_idx in range(num_layers):
        x = decoder_layer(x, weights, layer_idx, positions, CONFIG)
        if layer_idx < num_layers - 1:
            np.testing.assert_allclose(
                x, ref["hidden_states"][layer_idx + 1], atol=1e-2, rtol=1e-2,
                err_msg=f"prompt {prompt_idx}, layer {layer_idx} mismatch",
            )

    final_normed = rms_norm(x, weights["model.norm.weight"], CONFIG["rms_norm_eps"])
    np.testing.assert_allclose(
        final_normed, ref["hidden_states"][-1], atol=1e-2, rtol=1e-2,
        err_msg=f"prompt {prompt_idx}, final norm mismatch",
    )


@pytest.mark.parametrize("prompt_idx", range(len(TEST_PROMPTS)))
def test_final_logits_match_reference(weights, tokenized_prompts, reference_model, prompt_idx):
    input_ids = tokenized_prompts[prompt_idx]
    ref = run_reference(reference_model, input_ids)
    your_logits = forward(input_ids, weights, CONFIG)
    np.testing.assert_allclose(your_logits, ref["logits"], atol=ATOL)


@pytest.mark.parametrize("prompt_idx", range(len(TEST_PROMPTS)))
def test_greedy_next_token_matches_reference(weights, tokenized_prompts, reference_model, prompt_idx):
    """Sanity check in plain terms: does the actual predicted next token agree?"""
    input_ids = tokenized_prompts[prompt_idx]
    ref = run_reference(reference_model, input_ids)
    your_logits = forward(input_ids, weights, CONFIG)
    assert int(np.argmax(your_logits[-1])) == int(np.argmax(ref["logits"][-1]))
