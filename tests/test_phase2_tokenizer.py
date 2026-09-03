"""
Phase 2 verification: round-trips 10,000 strings through YOUR tokenizer and
compares token ids exactly against the reference (the standalone `tokenizers`
package - not `transformers`), used strictly as a verification oracle.

Run with:  python3 -m pytest tests/test_phase2_tokenizer.py -v
"""
import os
import random
import string

import pytest
from tokenizers import Tokenizer

from src.bpe_tokenizer import (
    load_byte_to_unicode,
    load_tokenizer_data,
    encode,
    decode,
)

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct", "tokenizer.json"
)


def generate_test_strings(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)

    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello",
        "world", "Python", "transformer", "attention", "is", "all", "you", "need",
        "café", "naïve", "résumé", "GPT-2", "tokenizer", "byte-pair", "encoding",
    ]
    unicode_chunks = ["日本語", "中文", "Русский", "العربية", "🚀", "😀", "🎉", "→", "™", "€", "½"]
    code_chunks = ["def foo(x):", "return x + 1", "self.value = None", "  \t  ", "a_b_c", "camelCase"]

    strings = []
    strings.append("")                     # empty string
    strings.append(" ")                    # single space
    strings.append("\n")                   # single newline
    strings.append("a")                    # single char

    for _ in range(n - len(strings)):
        kind = rng.random()
        if kind < 0.4:
            # a random "sentence" of english-ish words
            n_words = rng.randint(1, 15)
            sentence = " ".join(rng.choice(words) for _ in range(n_words))
            if rng.random() < 0.3:
                sentence = sentence.capitalize() + rng.choice([".", "!", "?", ""])
            strings.append(sentence)
        elif kind < 0.55:
            # numbers
            strings.append(str(rng.randint(-10**9, 10**9)))
        elif kind < 0.65:
            # decimals
            strings.append(f"{rng.uniform(-1e6, 1e6):.{rng.randint(0,6)}f}")
        elif kind < 0.75:
            # unicode / emoji mixes
            n_chunks = rng.randint(1, 5)
            strings.append(" ".join(rng.choice(unicode_chunks) for _ in range(n_chunks)))
        elif kind < 0.85:
            # code-like snippets
            n_chunks = rng.randint(1, 4)
            strings.append("\n".join(rng.choice(code_chunks) for _ in range(n_chunks)))
        elif kind < 0.93:
            # whitespace edge cases
            ws = rng.choice([" ", "  ", "\t", "\n", "\n\n", "   \n  "])
            core = " ".join(rng.choice(words) for _ in range(rng.randint(1, 4)))
            strings.append(ws + core + ws)
        else:
            # random ascii punctuation soup
            pool = string.ascii_letters + string.digits + string.punctuation + " "
            length = rng.randint(1, 80)
            strings.append("".join(rng.choice(pool) for _ in range(length)))

    return strings


@pytest.fixture(scope="module")
def reference():
    return Tokenizer.from_file(TOKENIZER_PATH)


@pytest.fixture(scope="module")
def yours():
    byte_encoder = load_byte_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    vocab, merge_ranks, pretokenizer_pattern = load_tokenizer_data(TOKENIZER_PATH)
    return {
        "byte_encoder": byte_encoder,
        "byte_decoder": byte_decoder,
        "vocab": vocab,
        "merge_ranks": merge_ranks,
        "pretokenizer_pattern": pretokenizer_pattern,
    }


@pytest.fixture(scope="module")
def test_strings():
    return generate_test_strings(10_000)


def test_encode_matches_reference_on_10000_strings(yours, reference, test_strings):
    mismatches = []
    for text in test_strings:
        your_ids = encode(
            text,
            yours["vocab"],
            yours["merge_ranks"],
            yours["byte_encoder"],
            yours["pretokenizer_pattern"],
        )
        ref_ids = reference.encode(text, add_special_tokens=False).ids
        if your_ids != ref_ids:
            mismatches.append((text, your_ids, ref_ids))
            if len(mismatches) >= 10:
                break

    assert not mismatches, "First mismatches (of possibly more):\n" + "\n".join(
        f"text={t!r}\n  yours={y}\n  ref={r}" for t, y, r in mismatches
    )


def test_decode_round_trips_on_10000_strings(yours, test_strings):
    mismatches = []
    for text in test_strings:
        ids = encode(
            text,
            yours["vocab"],
            yours["merge_ranks"],
            yours["byte_encoder"],
            yours["pretokenizer_pattern"],
        )
        recovered = decode(ids, yours["vocab"], yours["byte_decoder"])
        if recovered != text:
            mismatches.append((text, recovered))
            if len(mismatches) >= 10:
                break

    assert not mismatches, "First round-trip failures (of possibly more):\n" + "\n".join(
        f"original={o!r}\n  recovered={r!r}" for o, r in mismatches
    )
