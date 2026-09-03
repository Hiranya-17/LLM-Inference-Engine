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
- [ ] Phase 2 — BPE tokenization
- [ ] Phase 3 — Naive forward pass
- [ ] Phase 4 — KV cache
- [ ] Phase 5 — Sampling
- [ ] Phase 6 — Quantization

### Phase 1: Safetensors parsing

[`src/safetensors_reader.py`](src/safetensors_reader.py) implements the `.safetensors` format from the byte layout up: an 8-byte header length, a JSON header describing each tensor's name/dtype/shape/byte-offsets, followed by the raw tensor data. Includes a from-scratch BF16→FP32 upcast (bit-shifting, no floating point math) since NumPy has no native bfloat16 type.

Verified in [`tests/test_phase1_safetensors.py`](tests/test_phase1_safetensors.py) against the `safetensors` + `ml_dtypes` packages, used strictly as an independent reference oracle (never imported by the actual implementation): tensor names, shapes/dtypes, full byte-range coverage of the data blob, bit-exact BF16 upcast, and structural sanity against the model's `config.json`.

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
