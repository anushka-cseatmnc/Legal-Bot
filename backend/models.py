"""Explicitly provision model files for offline indexing and retrieval."""

import argparse

from huggingface_hub import snapshot_download

from backend.db import BGE_MODEL, EMBEDDING_MODEL, EMBEDDING_REVISIONS
from backend.retrieval import RERANKER, RERANKER_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--bge", action="store_true")
    args = parser.parse_args()
    models = [(EMBEDDING_MODEL, EMBEDDING_REVISIONS[EMBEDDING_MODEL])]
    if args.reranker:
        models.append((RERANKER, RERANKER_REVISION))
    if args.bge:
        models.append((BGE_MODEL, EMBEDDING_REVISIONS[BGE_MODEL]))
    for model, revision in models:
        snapshot_download(model, revision=revision, token=False,
                          allow_patterns=["*.json", "model.safetensors", "vocab.txt"])


if __name__ == "__main__":
    main()
