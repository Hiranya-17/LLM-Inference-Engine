"""
A from-scratch byte-level BPE tokenizer, matching Qwen2.5's tokenizer.json.

No `tokenizers` or `transformers` package - those are reference-oracle-only,
used in tests/, never here.

--- The byte-to-unicode trick ---

Raw bytes (0-255) include a lot of invisible/control characters (space,
newline, tab, etc). To keep merges.txt and vocab.json as plain readable
text, GPT-2-style tokenizers remap every possible byte to some *visible*
unicode character before doing anything else:

  - Bytes that are already "nice" printable characters (most of printable
    ASCII, plus a couple of printable Latin-1 ranges) map to themselves.
  - Every other byte (space, newline, control characters, etc.) gets
    remapped to an unused unicode codepoint starting at 256 upward.

This is why you saw "Ġ" in merges.txt - it's the stand-in character for
a raw space byte (0x20).

--- The full pipeline ---

encode(text) -> ids:
  1. Split text into rough chunks with the pretokenizer regex (so BPE
     merges never cross a chunk boundary).
  2. For each chunk: turn it into UTF-8 bytes, then remap every byte
     through the byte-to-unicode table above.
  3. Within each remapped chunk, repeatedly find the adjacent pair of
     pieces with the LOWEST rank in the merge table and fuse them into
     one piece. Stop when no pair in the chunk has a rank anymore.
  4. Look up each final piece in the vocab to get its integer id.

decode(ids) -> text reverses every step: ids -> token strings (reverse
vocab) -> concatenate -> reverse the byte-to-unicode map back to raw
bytes -> UTF-8 decode.
"""
from __future__ import annotations

import json

import regex


def load_byte_to_unicode() -> dict[int, str]:
    """
    Build the fixed 256-entry byte -> single-character mapping used by
    GPT-2-style byte-level BPE (this is a well-known, fixed algorithm -
    not something to reverse-engineer from the model files).

    Algorithm:
        1. Start with three ranges of bytes that are already nice and
           printable, so they map to themselves:
             - ord('!') to ord('~')   (33-126: printable ASCII)
             - ord('¡') to ord('¬')   (161-172: printable Latin-1)
             - ord('®') to ord('ÿ')   (174-255: printable Latin-1)
        2. Every OTHER byte value (0-32, 127-160, 173) is "ugly" -
           whitespace, control characters, etc. Each of these gets
           mapped to a new, unused unicode codepoint, counting up from
           256 (i.e. the first leftover byte maps to chr(256), the next
           to chr(257), and so on).

    Returns:
        dict mapping byte value (0-255) -> the single unicode character
        standing in for it.
    """
    mapping = {}
    next_codepoint = 256
    nice_bytes = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    for i in range(256):
      if i in nice_bytes:
        mapping[i] = chr(i)
      else:
        mapping[i] = chr(next_codepoint)
        next_codepoint += 1
    return mapping



def load_tokenizer_data(path: str):
    """
    Parse tokenizer.json and pull out the three things encode/decode need.

    Returns:
        (vocab, merge_ranks, pretokenizer_pattern)

        vocab: dict mapping token string -> integer id.
                 Found at json["model"]["vocab"].

        merge_ranks: dict mapping (left_piece, right_piece) -> rank (int,
                 lower = higher priority / merge earlier).
                 Found at json["model"]["merges"], a list of strings each
                 like "i n" (space-separated pair) - the position in that
                 list IS the rank.

        pretokenizer_pattern: the regex string used to chunk raw text.
                 Found at json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    """
    with open(path, "rb") as f:
        data = json.load(f)
        vocab = data["model"]["vocab"]
        merges_list = data["model"]["merges"]
        merge_ranks = {}
        for rank, line in enumerate(merges_list):
            left, right = line.split()          # 'a b' -> ('a', 'b')
            merge_ranks[(left, right)] = rank
        pretokenizer_pattern = data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    return vocab, merge_ranks, pretokenizer_pattern


def pretokenize(text: str, pattern: str) -> list[str]:
    """
    Split raw text into chunks using the pretokenizer regex.

    Must use the `regex` package (not stdlib `re`) - the pattern uses
    unicode property escapes like \\p{L} that stdlib re doesn't support.

    Hint: regex.findall(pattern, text)
    """
    return regex.findall(pattern, text)


def bpe_merge(chunk: str, merge_ranks: dict) -> list[str]:
    """
    Run the actual byte-pair merge loop on one already byte-remapped chunk
    (a string where every character is one "piece" to start with).

    Algorithm:
        1. Start with `pieces` = a list of each individual character in
           the chunk.
        2. Repeat:
             - Look at every adjacent pair (pieces[i], pieces[i+1]).
             - Among the pairs that exist as a key in merge_ranks, find
               the one with the LOWEST rank (highest priority).
             - If no adjacent pair exists in merge_ranks at all, stop.
             - Otherwise, merge that one pair into a single piece
               (string concatenation) and go back to step 2 with the
               updated list.
        3. Return the final list of pieces.
    """
    pieces = list(chunk)
    while True:
        adjacent_pairs = [(pieces[i], pieces[i+1]) for i in range(len(pieces) - 1)]
        candidates = [p for p in adjacent_pairs if p in merge_ranks]
        if not candidates:
            break
        best_pair = min(candidates, key=lambda p: merge_ranks[p])
        for i in range(len(pieces) - 1):
            if (pieces[i], pieces[i+1]) == best_pair:
                merge_at = i
                break
        pieces = pieces[:merge_at] + [pieces[merge_at] + pieces[merge_at+1]] + pieces[merge_at+2:]
    return pieces


def encode(text: str, vocab: dict, merge_ranks: dict, byte_encoder: dict, pretokenizer_pattern: str) -> list[int]:
    """
    Full text -> token ids pipeline. Wires together pretokenize,
    the byte remap, bpe_merge, and a vocab lookup.

    For each chunk from pretokenize(text, pretokenizer_pattern):
      1. Encode the chunk to UTF-8 bytes.
      2. Map each byte through byte_encoder to build the remapped string.
      3. Run bpe_merge on that remapped string.
      4. Look up each resulting piece in vocab, append its id.

    Return the concatenated list of ids for the whole text.
    """
    chunks = pretokenize(text, pretokenizer_pattern)
    ids = []
    for chunk in chunks:
        utf8_bytes = chunk.encode('utf-8')
        remapped = ''.join(byte_encoder[b] for b in utf8_bytes)
        final_list=bpe_merge(remapped, merge_ranks)
        for piece in final_list:
            ids.append(vocab[piece])
    return ids


def decode(ids: list[int], vocab: dict, byte_decoder: dict) -> str:
    """
    Full token ids -> text pipeline. Reverses encode().

    1. Build (once) the reverse of vocab: id -> token string.
    2. Look up and concatenate every id's token string.
    3. Walk the concatenated string one character at a time, using
       byte_decoder (the reverse of byte_encoder: character -> original
       byte value) to recover the original byte sequence.
    4. UTF-8 decode those bytes back into the original text.
    """
    reverse_vocab = {v: k for k, v in vocab.items()}
    remapped_string = ''.join(reverse_vocab[i] for i in ids)
    byte_values = [byte_decoder[ch] for ch in remapped_string]
    raw_bytes = bytes(byte_values)
    return raw_bytes.decode('utf-8')