# LLM Inference Engine From Scratch

A from-scratch implementation of an LLM inference engine, built to understand exactly how a transformer runs — no `transformers`, no `llama.cpp`, no `torch.generate()`. Every piece (file format, tokenizer, forward pass, KV cache, sampling, quantization) is implemented by hand and verified against a reference implementation before moving to the next stage.

**Target model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
**Target hardware:** Apple Silicon (CPU/NumPy)

## Approach

Each phase is built and independently verified for correctness before the next one starts:

1. **Safetensors parsing** — parse the model's weight file from the raw byte format, no `safetensors` library in the actual implementation.
2. **BPE tokenization** — round-trip tokenization checked against the reference tokenizer.
3. **Naive forward pass** — float32, no cache, greedy decoding, logits checked against reference within `1e-3`.
4. **KV cache** — proven identical output to the uncached version, with measured speedup.
5. **Sampling** — temperature, top-k, top-p, with a seedable RNG.
6. **Quantization** — INT8 then INT4, with perplexity measured at each level.

## Status

- [x] Phase 1 — Safetensors parsing
- [x] Phase 2 — BPE tokenization
- [x] Phase 3 — Naive forward pass
- [x] Phase 4 — KV cache
- [x] Phase 5 — Sampling
- [x] Phase 6 — Quantization (per-tensor)

### Phase 1: Safetensors parsing

[`src/safetensors_reader.py`](src/safetensors_reader.py) implements the `.safetensors` format from the byte layout up: an 8-byte header length, a JSON header describing each tensor's name/dtype/shape/byte-offsets, followed by the raw tensor data. Includes a from-scratch BF16→FP32 upcast (bit-shifting, no floating point math) since NumPy has no native bfloat16 type.

Verified in [`tests/test_phase1_safetensors.py`](tests/test_phase1_safetensors.py) against the `safetensors` + `ml_dtypes` packages, used strictly as an independent reference oracle (never imported by the actual implementation): tensor names, shapes/dtypes, full byte-range coverage of the data blob, bit-exact BF16 upcast, and structural sanity against the model's `config.json`.

### Phase 2: BPE tokenization

[`src/bpe_tokenizer.py`](src/bpe_tokenizer.py) implements Qwen2.5's byte-level BPE tokenizer from `tokenizer.json`'s raw contents: the GPT-2-style byte↔unicode remapping table, the regex-based pretokenizer, the iterative byte-pair merge loop, and both `encode()`/`decode()`. No `tokenizers` or `transformers` library in the actual implementation.

Verified in [`tests/test_phase2_tokenizer.py`](tests/test_phase2_tokenizer.py) against the standalone `tokenizers` package (not `transformers`), used strictly as a reference oracle: 10,000 generated strings (English sentences, numbers, unicode/emoji, code snippets, whitespace edge cases, random punctuation) all produce token ids identical to the reference, and all round-trip through `decode()` back to the original text exactly.

### Phase 3: Naive forward pass

[`src/forward_pass.py`](src/forward_pass.py) implements Qwen2.5's full transformer forward pass in NumPy: token embedding lookup, RMSNorm, rotary position embeddings (RoPE), grouped-query attention (14 query heads sharing 2 key/value heads) with causal masking, and a SwiGLU MLP block — 24 layers, each wrapped in residual connections, followed by a final norm and a tied-embedding output projection. No `torch` or `transformers` in the actual implementation.

Verified in [`tests/test_phase3_forward_pass.py`](tests/test_phase3_forward_pass.py) against `torch`+`transformers` (CPU, float32), used strictly as a reference oracle — prompts are tokenized with this project's own phase 2 tokenizer, and the reference is only used to compute ground-truth logits for those exact token ids. Across 3 test prompts: every one of the 24 layers' hidden states matches the reference, final logits match within `1e-3`, and the actual greedy-decoded next token agrees exactly.

### Phase 4: KV cache

[`src/kv_cache.py`](src/kv_cache.py) adds key/value caching on top of phase 3: instead of recomputing K/V for the entire growing sequence on every generation step, each layer caches every position's K/V and only computes it once, for the new token(s). One masking rule (query position `i` may attend to key position `j` whenever `j <= i`, using absolute sequence positions) handles both the initial full-prompt "prefill" and later single-token "decode" steps without special-casing either.

Verified in [`tests/test_phase4_kv_cache.py`](tests/test_phase4_kv_cache.py) against this project's own uncached phase 3/4 baseline (`generate_naive`) — no external reference needed, per the original spec. Across 2 prompts, `generate_with_cache` produces **token-for-token identical** output to `generate_naive`.

**Honest speedup number:** only ~1.1-1.2x at these scales (generating 10-40 tokens on a 0.5B model), not the large speedup a KV cache implies in principle. The cache is doing genuinely less work — confirmed by the speedup direction trending up as sequence length grows — but at such short sequences, wall-clock time here is dominated by fixed Python/NumPy overhead (a `for` loop over 14 attention heads and dict lookups by string key for every weight tensor, per layer, per token) rather than by the O(n²)-vs-O(n) algorithmic difference the cache actually fixes. That gap would widen substantially at longer sequences or with a vectorized (rather than per-head Python loop) implementation — out of scope for this phase, which asked for correctness plus a measurement, not an optimized implementation.

### Phase 5: Sampling

[`src/sampling.py`](src/sampling.py) implements temperature scaling, top-k filtering, top-p (nucleus) filtering, softmax, and a full sampling pipeline (`sample()`) that chains them together and draws a token id from a seedable `numpy.random.Generator`. Top-k and top-p both work by setting excluded logits to `-inf` (so they get probability 0 after softmax) rather than by removing entries — this keeps every function's output the same shape as its input, which is what lets them chain: `apply_temperature` → `top_k_filter` → `top_p_filter` → `softmax` → sample.

Verified in [`tests/test_phase5_sampling.py`](tests/test_phase5_sampling.py): `apply_temperature`, `top_k_filter`, and `top_p_filter` are each checked against `transformers`' own `TemperatureLogitsWarper`/`TopKLogitsWarper`/`TopPLogitsWarper` classes (used strictly as a reference oracle, never imported by the actual implementation) across multiple parameter values. `softmax` is checked against `torch.softmax`. Since the actual random draw can't be checked against a reference the same way (independent RNGs draw different random numbers even from identical probabilities), `sample()` is instead verified statistically — 100,000 draws from a known 5-element distribution match the expected probabilities within `0.01` — and for reproducibility: the same seed produces the exact same sequence of draws, and `top_k=1` always agrees with greedy argmax regardless of seed.

### Phase 6: Quantization (per-tensor)

[`src/quantization.py`](src/quantization.py) implements affine (symmetric, per-tensor) INT8 and INT4 quantization: for each 2D weight matrix, one scale factor (`max(abs(tensor)) / qmax`) is computed for the *entire tensor*, every value is rounded to the nearest representable integer and clipped to range, then immediately dequantized back to float32 ("fake quantization" — this measures accuracy impact, not memory savings; a real deployment would keep the integers packed and use dedicated low-bit matmul kernels, which is out of scope here). RMSNorm weights (1D) are left untouched.

Verified in [`tests/test_phase6_quantization.py`](tests/test_phase6_quantization.py) two ways: unit tests check the quantization math directly against its own definition (the maximum-magnitude element always maps to exactly ±qmax, dequantized error never exceeds half a scale step), and an end-to-end test measures perplexity on a short paragraph at fp32, INT8, and INT4.

**Honest result — per-tensor INT4 breaks the model:**

| precision | perplexity |
|---|---|
| fp32 | 9.9 |
| INT8 | 10.4 |
| INT4 | ~5.5 × 10^8 |

INT8 barely hurts (127 representable levels per tensor is plenty of resolution). INT4 per-tensor doesn't just degrade the model, it destroys it. The reason: **one scale factor has to cover an entire weight matrix.** If even a single weight in that matrix is a large outlier, the scale stretches to accommodate it — and with only 7 representable levels on each side of zero, every other (normal-sized) weight in that tensor gets crushed toward zero. One outlier wrecks the whole tensor. This is a well-known real failure mode of naive per-tensor quantization, not a bug in this implementation — and it's the direct motivation for per-row quantization (a separate scale per row, so one outlier only damages its own row), which is the next iteration of this phase.

## Setup

```bash
git clone <this-repo-url>
cd llm-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy only, what src/ actually needs
pip install -r requirements-dev.txt    # + pytest, safetensors, ml_dtypes for tests

python3 src/download_model.py          # fetches the model files from Hugging Face
```

## Running

```bash
source .venv/bin/activate

python3 -m pytest tests/ -v            # run all verification tests
python3 -m src.inspect_model           # print every tensor's name/dtype/shape
python3 -m bench.benchmark             # show the tokens/sec log across phases
```
