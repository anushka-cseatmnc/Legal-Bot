"""Shared embedding and index configuration for ingestion and retrieval."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISIONS = {
    EMBEDDING_MODEL: EMBEDDING_REVISION,
    BGE_MODEL: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
}
EMBEDDING_DIMENSION = 384
COLLECTION_NAME = "legalbot"
INDEX_SCHEMA = 1
ACTIVE_INDEX_FILE = "active-index.json"


class IndexUnavailable(RuntimeError):
    """The published index is absent or incompatible."""


def project_path(value: str | Path) -> Path:
    return (PROJECT_ROOT / value).resolve()


def index_root() -> Path:
    return project_path(os.getenv("CHROMA_DB_PATH") or "chroma_db")


@lru_cache(maxsize=2)
def get_embeddings(model: str = EMBEDDING_MODEL):
    from langchain_huggingface import HuggingFaceEmbeddings

    query_options = {}
    encode_options = {"batch_size": 32}
    if model == BGE_MODEL:
        encode_options["normalize_embeddings"] = True
        query_options = {"prompt": "Represent this sentence for searching relevant passages: ",
                         "normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu", "revision": EMBEDDING_REVISIONS[model],
                      "local_files_only": True, "trust_remote_code": False},
        encode_kwargs=encode_options,
        query_encode_kwargs=query_options,
        show_progress=False,
    )


def read_index_manifest(root: Path, model: str = EMBEDDING_MODEL) -> dict:
    try:
        manifest = json.loads((root / ACTIVE_INDEX_FILE).read_text(encoding="utf-8"))
        if (
            manifest["schema"] == INDEX_SCHEMA
            and manifest["embedding_model"] == model
            and manifest["embedding_revision"] == EMBEDDING_REVISIONS[model]
            and manifest["dimension"] == EMBEDDING_DIMENSION
            and manifest["collection"] == COLLECTION_NAME
            and type(manifest["chunks"]) is int
            and manifest["chunks"] > 0
            and re.fullmatch(r"[0-9a-f]{32}", manifest["snapshot"])
        ):
            return manifest
    except (OSError, ValueError, KeyError, TypeError):
        pass
    raise IndexUnavailable(f"No compatible index at {root}. Run python -m backend.ingest.")


def connect_snapshot(path: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def validate_collection(path: Path, expected_count: int, model: str = EMBEDDING_MODEL) -> chromadb.ClientAPI:
    if not (path / "chroma.sqlite3").is_file():
        raise IndexUnavailable(f"Index snapshot is missing at {path}. Rebuild the index.")
    client = connect_snapshot(path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError as error:
        raise IndexUnavailable("Index snapshot has no LegalBot collection.") from error
    metadata = collection.metadata or {}
    if (
        collection.count() != expected_count
        or metadata.get("embedding_revision") != EMBEDDING_REVISIONS[model]
        or metadata.get("index_schema") != INDEX_SCHEMA
    ):
        raise IndexUnavailable("Index snapshot failed validation. Rebuild the index.")
    vectors = collection.get(limit=1, include=["embeddings"])["embeddings"]
    if vectors is None or len(vectors) != 1 or len(vectors[0]) != EMBEDDING_DIMENSION:
        raise IndexUnavailable("Index snapshot has incompatible embedding dimensions. Rebuild the index.")
    return client


@lru_cache(maxsize=1)
def _open_collection(path: Path, expected_count: int, model: str) -> Chroma:
    return Chroma(
        client=validate_collection(path, expected_count, model),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(model),
        create_collection_if_not_exists=False,
    )


def get_vector_store(root: Path | None = None, model: str = EMBEDDING_MODEL) -> Chroma:
    root = index_root() if root is None else root
    manifest = read_index_manifest(root, model)
    return _open_collection(root / "snapshots" / manifest["snapshot"], manifest["chunks"], model)
