"""
Prints every tensor in the model: name, dtype, shape, and size.
Uses your safetensors_reader.py - this is just the reporting layer on top.
"""
import os
import time

from src.safetensors_reader import read_header
from bench.benchmark import record

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct", "model.safetensors")

DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


def num_elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    t0 = time.perf_counter()
    tensors, data_start = read_header(MODEL_PATH)
    elapsed = time.perf_counter() - t0

    total_params = 0
    total_bytes = 0

    print(f"{'name':<55} {'dtype':<6} {'shape':<20} {'bytes':>12}")
    print("-" * 97)
    for name, info in sorted(tensors.items()):
        n_elem = num_elements(info.shape)
        n_bytes = n_elem * DTYPE_SIZES.get(info.dtype, 0)
        total_params += n_elem
        total_bytes += n_bytes
        print(f"{name:<55} {info.dtype:<6} {str(info.shape):<20} {n_bytes:>12,}")

    file_size = os.path.getsize(MODEL_PATH)
    print("-" * 97)
    print(f"tensors: {len(tensors)}")
    print(f"total parameters: {total_params:,}")
    print(f"total tensor bytes: {total_bytes:,}")
    print(f"header size: {data_start} bytes")
    print(f"file size on disk: {file_size:,} bytes")
    print(f"header parse time: {elapsed*1000:.2f} ms")

    record(
        phase="phase1_safetensors_parse",
        tokens=0,
        elapsed_s=elapsed,
        extra={
            "num_tensors": len(tensors),
            "total_params": total_params,
            "total_bytes": total_bytes,
            "file_size": file_size,
        },
    )


if __name__ == "__main__":
    main()
