"""
Downloads Qwen2.5-0.5B-Instruct files straight from the Hugging Face CDN.

This is plain file fetching, not part of the "implement it yourself" rule -
we're not using `transformers` or any model-loading library, just pulling
raw bytes over HTTP with the standard library.
"""
import os
import sys
import urllib.request

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

DEST = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct")


def download(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f"  skip (exists, {size:,} bytes): {os.path.basename(dest_path)}")
        return

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            sys.stdout.write(f"\r  {os.path.basename(dest_path)}: {pct}% ({downloaded:,}/{total_size:,} bytes)")
            sys.stdout.flush()

    tmp_path = dest_path + ".part"
    urllib.request.urlretrieve(url, tmp_path, reporthook=report)
    os.rename(tmp_path, dest_path)
    print()


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading {REPO} into {os.path.abspath(DEST)}\n")
    for fname in FILES:
        url = f"{BASE_URL}/{fname}"
        dest_path = os.path.join(DEST, fname)
        download(url, dest_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
