"""Export existing vectors and restore them into Chroma on a fresh host."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from time import monotonic
from urllib.parse import urljoin, urlsplit
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np
import requests
from filelock import FileLock, Timeout

from backend.db import (
    ACTIVE_INDEX_FILE, COLLECTION_NAME, EMBEDDING_DIMENSION, EMBEDDING_MODEL,
    EMBEDDING_REVISION, INDEX_SCHEMA, PROJECT_ROOT, IndexUnavailable,
    connect_snapshot, index_root, project_path, read_index_manifest, validate_collection,
)

FILES = {
    "manifest.json", "documents.jsonl", "vectors.npy", "minilm.onnx", "tokenizer.json",
    "MiniLM-APACHE-2.0.txt", "MiniLM-NOTICE.txt",
}
MAX_ARCHIVE = 160 * 2**20
MAX_UNPACKED = 192 * 2**20
BATCH = 512
LOCAL_BUNDLE = "deployment_artifacts/legalbot-index.zip"


def sha256(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
        return digest.hexdigest()


def export_bundle(root, output):
    from huggingface_hub import hf_hub_download

    manifest = read_index_manifest(root)
    snapshot = root / "snapshots" / manifest["snapshot"]
    client = validate_collection(snapshot, manifest["chunks"])
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    pdfs = {path.name: sha256(path) for path in (PROJECT_ROOT / "data").glob("*.pdf")}
    if set(pdfs) != set(manifest["sources"]):
        raise IndexUnavailable("Only the bundled public corpus can be exported.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        with (workspace / "documents.jsonl").open("w", encoding="utf-8") as rows:
            vectors = np.lib.format.open_memmap(workspace / "vectors.npy", mode="w+", dtype="<f4",
                                                 shape=(manifest["chunks"], EMBEDDING_DIMENSION))
            for offset in range(0, manifest["chunks"], BATCH):
                part = collection.get(offset=offset, limit=BATCH, include=["documents", "metadatas", "embeddings"])
                for key, document, metadata in zip(part["ids"], part["documents"], part["metadatas"]):
                    name = metadata.get("document")
                    if name not in pdfs or metadata.get("document_sha256") != pdfs[name] or metadata.get("source") != name:
                        raise IndexUnavailable("Index contains material outside the bundled corpus.")
                    rows.write(json.dumps({"id": key, "document": document, "metadata": metadata}, ensure_ascii=False) + "\n")
                vectors[offset:offset + len(part["ids"])] = part["embeddings"]
            vectors.flush()
            del vectors
        (workspace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        def model_file(name):
            return hf_hub_download(EMBEDDING_MODEL, name, revision=EMBEDDING_REVISION, token=False)
        files = {name: workspace / name for name in ("manifest.json", "documents.jsonl", "vectors.npy")}
        files.update({"minilm.onnx": model_file("onnx/model.onnx"), "tokenizer.json": model_file("tokenizer.json")})
        pending = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            with ZipFile(pending, "w", compression=ZIP_DEFLATED, compresslevel=5) as archive:
                for name, path in files.items():
                    archive.write(path, name)
                archive.write(PROJECT_ROOT / "data" / "MiniLM-APACHE-2.0.txt", "MiniLM-APACHE-2.0.txt")
                archive.writestr("MiniLM-NOTICE.txt",
                                 f"{EMBEDDING_MODEL}\n"
                                 f"https://huggingface.co/{EMBEDDING_MODEL}/tree/{EMBEDDING_REVISION}\n"
                                 "The unmodified ONNX graph and tokenizer are distributed under Apache-2.0.\n")
            if pending.stat().st_size > MAX_ARCHIVE:
                raise IndexUnavailable("Index bundle exceeds the configured size limit.")
            pending.replace(output)
        finally:
            pending.unlink(missing_ok=True)
    return sha256(output)


def public_release_url(url):
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and parsed.netloc == "github.com" and not parsed.query
            and not parsed.fragment and re.fullmatch(r"/[^/]+/[^/]+/releases/download/[^/]+/[^/]+", parsed.path))


def download_bundle(url, destination):
    if not public_release_url(url):
        raise IndexUnavailable("Use the asset URL of a public GitHub release.")
    hosts = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
    deadline = monotonic() + 120
    with requests.Session() as session:
        for _ in range(4):
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.netloc not in hosts:
                break
            with session.get(url, timeout=(8, 30), stream=True, allow_redirects=False) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                with destination.open("wb") as output:
                    size = 0
                    for block in response.iter_content(2**20):
                        size += len(block)
                        if monotonic() >= deadline:
                            raise IndexUnavailable("The index release download timed out.")
                        if size > MAX_ARCHIVE:
                            raise IndexUnavailable("Index bundle exceeds the configured size limit.")
                        output.write(block)
                return
    raise IndexUnavailable("The index release download redirected unexpectedly.")


def hydrate_index(root, files):
    manifest = json.loads((files / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("embedding_model") != EMBEDDING_MODEL or manifest.get("schema") != INDEX_SCHEMA
            or manifest.get("embedding_revision") != EMBEDDING_REVISION
            or manifest.get("dimension") != EMBEDDING_DIMENSION
            or manifest.get("collection") != COLLECTION_NAME
            or type(manifest.get("chunks")) is not int or not 0 < manifest["chunks"] <= 100_000):
        raise IndexUnavailable("The index bundle is incompatible.")
    size = manifest["chunks"]
    vectors = np.load(files / "vectors.npy", allow_pickle=False, mmap_mode="r")
    if vectors.shape != (size, EMBEDDING_DIMENSION) or vectors.dtype != np.dtype("<f4") or not np.isfinite(vectors).all():
        raise IndexUnavailable("The index bundle contains invalid vectors.")
    snapshot = uuid4().hex
    directory = root / "snapshots" / snapshot
    directory.mkdir(parents=True)
    client = connect_snapshot(directory)
    collection = client.create_collection(COLLECTION_NAME, metadata={
        "embedding_revision": EMBEDDING_REVISION, "index_schema": INDEX_SCHEMA,
    }, embedding_function=None)
    with (files / "documents.jsonl").open(encoding="utf-8") as source:
        for offset in range(0, size, BATCH):
            rows = [json.loads(source.readline()) for _ in range(min(BATCH, size - offset))]
            collection.add(ids=[row["id"] for row in rows],
                           documents=[row["document"] for row in rows],
                           metadatas=[row["metadata"] for row in rows],
                           embeddings=vectors[offset:offset + len(rows)])
        if source.read(1):
            raise IndexUnavailable("The index bundle contains unexpected rows.")
    del vectors
    validate_collection(directory, size)
    for name in ("minilm.onnx", "tokenizer.json"):
        shutil.copyfile(files / name, directory / name)
    manifest["snapshot"] = snapshot
    pending = root / f"{snapshot}.json"
    pending.write_text(json.dumps(manifest), encoding="utf-8")
    pending.replace(root / ACTIVE_INDEX_FILE)
    return directory, size


def prepare_bundle(digest, *, local=None, url=None, cache_dir=None):
    if not local and not url:
        candidate = project_path(LOCAL_BUNDLE)
        if candidate.is_file():
            local = candidate
    if local:
        local = project_path(local)
        if not digest:
            try:
                digest = local.with_suffix(".sha256").read_text(encoding="ascii").strip().lower()
            except (OSError, UnicodeError) as exc:
                raise IndexUnavailable("Could not read the local index bundle's SHA-256 sidecar.") from exc
    if not re.fullmatch(r"[a-f0-9]{64}", digest) or not (local or url):
        raise IndexUnavailable("Configure an index bundle and its SHA-256 checksum.")
    cache = Path(cache_dir or tempfile.gettempdir()) / "legalbot-index"
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / digest
    root.mkdir(exist_ok=True)
    try:
        with FileLock(cache / f"{digest}.lock", timeout=120):
            if (root / ACTIVE_INDEX_FILE).is_file():
                manifest = read_index_manifest(root)
                snapshot = root / "snapshots" / manifest["snapshot"]
                validate_collection(snapshot, manifest["chunks"])
                if not all((snapshot / name).is_file() for name in ("minilm.onnx", "tokenizer.json")):
                    raise IndexUnavailable("The cached index bundle is incomplete.")
                return snapshot, manifest["chunks"]
            with tempfile.TemporaryDirectory(dir=root) as directory:
                workspace = Path(directory)
                archive_path = local if local else workspace / "bundle.zip"
                if not local:
                    download_bundle(url, archive_path)
                if archive_path.stat().st_size > MAX_ARCHIVE or sha256(archive_path) != digest:
                    raise IndexUnavailable("The index bundle checksum or size is incorrect.")
                with ZipFile(archive_path) as archive:
                    entries = archive.infolist()
                    if (len(entries) != len(FILES) or {item.filename for item in entries} != FILES
                            or sum(item.file_size for item in entries) > MAX_UNPACKED):
                        raise IndexUnavailable("The index bundle has an invalid layout.")
                    for item in entries:
                        with archive.open(item) as source, (workspace / item.filename).open("wb") as output:
                            shutil.copyfileobj(source, output)
                return hydrate_index(root, workspace)
    except (BadZipFile, OSError, ValueError, KeyError, requests.RequestException, Timeout) as exc:
        raise IndexUnavailable("Could not initialize the published legal index.") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = export_bundle(index_root(), args.output)
    args.output.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    print(f"{args.output}  {digest}")


if __name__ == "__main__":
    main()
