"""
A from-scratch safetensors reader. No `safetensors` package, no torch.

YOUR TASK: implement the three functions below. Everything you need is in
the docstrings. Don't peek at the real `safetensors` library source - the
whole point is to derive this from the format spec.

--- The format, in full ---

A .safetensors file is exactly three things back to back:

  [0:8]     little-endian uint64 N = byte length of the header
  [8:8+N]   UTF-8 JSON header (see shape below)
  [8+N:]    raw tensor bytes, concatenated, no separators, no padding

Header JSON shape:

    {
      "<tensor name>": {
        "dtype": "BF16" | "F32" | "F16" | "I64" | ...,
        "shape": [dim0, dim1, ...],
        "data_offsets": [start, end]   # BYTE offsets, relative to the
                                        # start of the data blob (i.e. NOT
                                        # relative to the start of the file -
                                        # you must add 8+N yourself)
      },
      ...
      "__metadata__": { ... }   # optional, not a tensor, skip it
    }

`end - start` for a tensor must equal:
    num_elements(shape) * dtype_size_in_bytes

There is no compression and no pickling. This format cannot execute code
when loaded - that's its entire reason for existing over PyTorch's old
pickle-based .bin checkpoints.

--- BF16 gotcha ---

Qwen2.5 ships its weights as BF16. NumPy has no native bfloat16 dtype.
But bfloat16 is *just the top 16 bits of a float32* (1 sign + 8 exponent +
7 mantissa bits, vs float32's 1+8+23). So to upcast a bf16 buffer to
float32 losslessly:

  1. Read the raw bytes as uint16.
  2. Widen each uint16 to uint32.
  3. Left-shift each by 16 bits (this places the bf16 bits into the *upper*
     16 bits of the 32-bit word, which is exactly where they'd sit in a
     real float32 - the lower 16 mantissa bits become zero).
  4. Reinterpret (view, not cast) that uint32 buffer as float32.

No floating point math needed anywhere in that - it's pure bit shuffling.
"""
from __future__ import annotations

import json
import struct
import numpy as np
from dataclasses import dataclass


@dataclass
class TensorInfo:
    name: str
    dtype: str          # the raw string from the header, e.g. "BF16"
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]   # relative to the data blob, NOT the file


def read_header(path: str) -> tuple[dict[str, TensorInfo], int]:
    """
    Read and parse a safetensors file's header.

    Returns:
        (tensors, data_start_offset)

        tensors: dict mapping tensor name -> TensorInfo. Must NOT include
                 the "__metadata__" key (that's not a tensor).
        data_start_offset: the absolute byte offset in the file where the
                 raw tensor data blob begins (i.e. 8 + N).

    Hints:
        - struct.unpack("<Q", ...) reads a little-endian uint64.
        - Only read the first 8 bytes, then exactly N more bytes - don't
          read the whole (potentially multi-GB) file into memory here.
    """
    with open(path, "rb") as f:
      length_bytes = f.read(8)
      n = struct.unpack("<Q", length_bytes)[0]
      header_bytes = f.read(n)
      header = json.loads(header_bytes)

    tensors = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        tensors[key] = TensorInfo(
            name=key,
            dtype=value["dtype"],
            shape=tuple(value["shape"]),
            data_offsets=tuple(value["data_offsets"]),
        )

    return tensors, n + 8

def bf16_bytes_to_fp32(raw: bytes, shape: tuple[int, ...]):
    """
    Convert raw BF16 bytes into a numpy float32 array of the given shape.

    Must return an array with the exact values you'd get from casting the
    original bf16 numbers to float32 - not an approximation. See the
    module docstring for the bit-shift trick.

    Hint: np.frombuffer(raw, dtype=np.uint16) to get started.
    """
    rawbf16 = np.frombuffer(raw, dtype=np.uint16)
    widened = np.uint32(rawbf16)
    shifted = widened << np.uint32(16)
    as_float = shifted.view(np.float32)
    tensor_shape = as_float.reshape(shape)
    return tensor_shape



def load_tensor(path: str, info: TensorInfo, data_start_offset: int):
    """
    Load a single tensor's actual values from disk as a numpy array with
    the correct shape and (upcast-to-float32) dtype.

    Must open the file and read/seek only the bytes belonging to this one
    tensor (info.data_offsets, shifted by data_start_offset) - never load
    the whole file to get one tensor.

    For now, only BF16 needs to be supported (that's all this checkpoint
    uses) - but write it so adding F32/F16/I64 later is a one-line addition
    per dtype, not a rewrite.
    """
    bytes_read = info.data_offsets[1] - info.data_offsets[0]
    start = data_start_offset + info.data_offsets[0]
    with open(path, "rb") as f:
        f.seek(start)
        rest = f.read(bytes_read)
        return bf16_bytes_to_fp32(rest, info.shape)