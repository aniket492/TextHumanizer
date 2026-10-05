"""Pre-download the rewrite and similarity models into the Hugging Face cache (resumable).

Usage:  python scripts/download_model.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--retries 8]
Reads MODEL_NAME / SIMILARITY_MODEL from the environment when flags are omitted.
"""
from __future__ import annotations

import argparse
import os
import time

import _common  # noqa: F401

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def fetch(repo_id: str, retries: int) -> str:
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.py", "modules.json", "1_Pooling/*"]
    for attempt in range(1, retries + 1):
        try:
            path = snapshot_download(repo_id, allow_patterns=patterns)
            print(f"OK  {repo_id} -> {path}")
            return path
        except Exception as exc:  # network hiccups: resume on the next attempt
            wait = min(60, 5 * attempt)
            print(f"Attempt {attempt}/{retries} for {repo_id} failed ({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"Could not download {repo_id} after {retries} attempts.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--similarity-model", default=os.getenv("SIMILARITY_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    for repo in filter(None, [args.similarity_model, args.model]):
        fetch(repo, args.retries)


if __name__ == "__main__":
    main()
