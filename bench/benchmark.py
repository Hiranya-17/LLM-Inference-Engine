"""
Benchmark harness, shared across every phase of this project.

Usage from any phase's code:

    from bench.benchmark import record

    t0 = time.perf_counter()
    ... do the thing (parse a file, run a forward pass, generate N tokens) ...
    elapsed = time.perf_counter() - t0

    record(phase="phase1_safetensors_parse", tokens=0, elapsed_s=elapsed,
           extra={"num_tensors": 291, "total_bytes": 988097824})

Results are appended as one JSON object per line to bench/results.jsonl so the
log is append-only and diffable. Run this file directly to print a summary
table of everything logged so far.
"""
from __future__ import annotations

import json
import os
import platform
import time

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.jsonl")


def _hardware_tag() -> str:
    return f"{platform.system()} {platform.machine()} | {platform.processor() or platform.machine()}"


def record(phase: str, tokens: int, elapsed_s: float, extra: dict | None = None) -> dict:
    """
    Append one benchmark result. `tokens` is tokens *generated* (0 if this
    phase has no generation yet, e.g. phase 1's file parsing). tokens/sec is
    computed as tokens / elapsed_s, or None if tokens == 0.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": phase,
        "tokens": tokens,
        "elapsed_s": round(elapsed_s, 6),
        "tokens_per_sec": round(tokens / elapsed_s, 3) if tokens and elapsed_s > 0 else None,
        "hardware": _hardware_tag(),
    }
    if extra:
        entry["extra"] = extra

    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def load_all() -> list[dict]:
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def report() -> None:
    rows = load_all()
    if not rows:
        print("No benchmark results yet. Run a phase's script first.")
        return

    print(f"{'phase':<30} {'tokens':>8} {'elapsed_s':>10} {'tok/s':>10}  timestamp")
    print("-" * 80)
    for r in rows:
        tps = f"{r['tokens_per_sec']:.2f}" if r["tokens_per_sec"] is not None else "-"
        print(f"{r['phase']:<30} {r['tokens']:>8} {r['elapsed_s']:>10.4f} {tps:>10}  {r['timestamp']}")


if __name__ == "__main__":
    report()
