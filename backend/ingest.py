"""Build and publish a validated local Chroma index from the legal corpus."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__ in (None, ""):
    # Keep the original direct-script command working.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db import (
    ACTIVE_INDEX_FILE,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_REVISIONS,
    INDEX_SCHEMA,
    PROJECT_ROOT,
    IndexUnavailable,
    connect_snapshot,
    get_embeddings,
    index_root,
    project_path,
    read_index_manifest,
    validate_collection,
)

logger = logging.getLogger(__name__)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
INDEX_BATCH = 128
GAZETTE_HEADING = re.compile(r'(?m)^[ \t]*(\d{1,3})\.[ \t]*(?=\([0-9a-z]+\)|[A-Z“"])')
TITLED_HEADING = re.compile(
    r"(?m)^[ \t]*(\d{1,3}[A-Z]{0,3})\.[ \t]+([^\n]{1,150}?)\.[ \t]*[—–]"
)
ARTICLE_HEADING = re.compile(
    r"(?m)^[ \t]*(\d{1,3}[A-Z]{0,3})\.[ \t]+([^\n]{1,125}?(?:\n[ \t]*[a-z][^\n]{0,95}?)?)\.[ \t]*[—–]"
)
SCHEDULE = re.compile(r"(?m)^[ \t]*THE (?:FIRST |SECOND )?SCHEDULE\b")
BNS_EXCEPTION = re.compile(r"(?im)^[ \t]*\(2\)[ \t]+Whoever causes death\b")
BNSS_EXCEPTION = re.compile(r"(?m)^[ \t]*106[ \t]*\(2\)[ \t]+Causing death\b")


class ProvisionTracker:
    """Keep page-local text while tracking sections that span page breaks."""

    def __init__(self, profile: dict):
        self.act_id = profile.get("act_id")
        self.last_expected = profile.get("gazette_sections")
        self.first_body_page = profile.get("body_page")
        self.last_body_page = profile.get("last_body_page")
        self.headings = profile.get("section_headings", {})
        self.titled_heading = ARTICLE_HEADING if profile.get("provision_type") == "article" else TITLED_HEADING
        self.section = None
        self.heading = None
        self.last_gazette_section = 0
        self.in_schedule = False
        self.in_bns_exception = False
        self.found_exception = False

    def segments(self, text: str, pdf_page: int):
        if self.first_body_page and pdf_page < self.first_body_page:
            yield text, {"authority_type": "front_matter"}
            return

        if self.last_body_page and pdf_page > self.last_body_page:
            yield self.schedule_segment(text)
            return

        if self.in_schedule:
            yield self.schedule_segment(text)
            return

        schedule = SCHEDULE.search(text) if self.last_expected else None
        body = text[:schedule.start()] if schedule else text
        if self.last_expected or self.first_body_page:
            headings = GAZETTE_HEADING if self.last_expected else self.titled_heading
            start = 0
            for match in headings.finditer(body):
                number = match[1]
                if self.last_expected:
                    if int(number) != self.last_gazette_section + 1:
                        continue
                    if int(number) > self.last_expected:
                        break
                    self.last_gazette_section = int(number)
                yield from self.section_segments(body[start:match.start()])
                self.section = number
                self.heading = self.headings.get(number) if self.last_expected else " ".join(match[2].split())
                self.in_bns_exception = False
                start = match.start()
            yield from self.section_segments(body[start:])
        else:
            yield body, {}

        if schedule:
            self.in_schedule = True
            yield self.schedule_segment(text[schedule.start():])

    def section_segments(self, text: str):
        metadata = {"section": self.section} if self.section else {}
        if self.heading:
            metadata["section_heading"] = self.heading
        if self.last_expected and not self.section:
            metadata["authority_type"] = "front_matter"
        if self.act_id == "BNS" and self.section == "106":
            exception = BNS_EXCEPTION.search(text)
            if exception:
                yield text[:exception.start()], metadata
                text = text[exception.start():]
                self.in_bns_exception = self.found_exception = True
            if self.in_bns_exception:
                metadata = {
                    **metadata,
                    "subsection": "2",
                    "legal_status": "excluded_at_commencement",
                }
        yield text, metadata

    def schedule_segment(self, text: str):
        metadata = {"authority_type": "schedule"}
        # A table page can mix in-force entries and the uncommenced entry.
        if self.act_id == "BNSS" and BNSS_EXCEPTION.search(text):
            self.found_exception = True
            metadata.update(
                legal_status="mixed_commencement",
                commencement_exception="First Schedule entry for BNS section 106(2)",
            )
        return text, metadata

    def validate(self):
        if self.last_expected and self.last_gazette_section != self.last_expected:
            raise ValueError(
                f"detected sections 1–{self.last_gazette_section}, expected 1–{self.last_expected}"
            )
        if self.act_id in {"BNS", "BNSS"} and self.last_expected and not self.found_exception:
            raise ValueError("could not locate the provision excluded at commencement")


def corpus_inventory(data_path: Path):
    manifest_path = data_path / "corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    files = sorted(
        (path for path in data_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    if not files:
        raise ValueError(f"No PDFs found in {data_path}")
    missing = set(manifest) - {path.name for path in files}
    if missing:
        raise ValueError(f"Missing declared PDFs: {', '.join(sorted(missing))}")
    inventory = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest.get(path.name, {}).get("sha256")
        if expected and digest != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")
        inventory[path.name] = digest
    return files, manifest, inventory


def extract_chunks(path: Path, profile: dict, digest: str, splitter):
    controls = {"sha256", "gazette_sections", "body_page", "last_body_page", "section_headings"}
    metadata = {
        "authority_type": "unclassified",
        "legal_status": "unverified",
        "jurisdiction": "IN",
        **{key: value for key, value in profile.items() if key not in controls},
        "source": path.name,
        "document": path.name,
        "document_sha256": digest,
    }
    tracker = ProvisionTracker(profile)
    chunks = []
    empty_pages = []
    page_count = 0
    for page in PyPDFLoader(str(path)).lazy_load():
        page_count += 1
        page_index = page.metadata["page"]
        page_number = page_index + 1
        page_meta = {
            **metadata,
            "page": page_index,
            "pdf_page": page_number,
            "page_label": page.metadata.get("page_label") or str(page_number),
        }
        if not page.page_content.strip():
            empty_pages.append(page_number)
            continue
        for text, provision in tracker.segments(page.page_content, page_number):
            chunk_meta = {**page_meta, **provision}
            if chunk_meta["legal_status"] in {"excluded_at_commencement", "mixed_commencement"}:
                chunk_meta.pop("effective_from", None)
            for ordinal, content in enumerate(splitter.split_text(text)):
                chunk_index = len(chunks)
                chunk_id = hashlib.sha256(
                    f"{path.name}\0{digest}\0{chunk_index}".encode("utf-8")
                ).hexdigest()
                chunks.append(Document(
                    id=chunk_id,
                    page_content=content,
                    metadata={**chunk_meta, "chunk_id": chunk_id,
                              "document_order": chunk_index, "segment_order": ordinal},
                ))
    tracker.validate()
    if not chunks:
        raise ValueError("no extractable text")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("PDF changed during extraction")
    return chunks, {"pages": page_count, "empty_pages": empty_pages, "chunks": len(chunks)}


def validate_snapshot(collection, documents, source_stats):
    if collection.count() != len(documents):
        raise RuntimeError("Chroma count differs from the extracted chunk count")
    verified_sources = {}
    for offset in range(0, len(documents), INDEX_BATCH):
        batch = documents[offset:offset + INDEX_BATCH]
        expected = {doc.id: doc.metadata for doc in batch}
        stored = collection.get(ids=list(expected), include=["metadatas"])
        if dict(zip(stored["ids"], stored["metadatas"])) != expected:
            raise RuntimeError("Chroma did not preserve the chunk metadata")
        for metadata in stored["metadatas"]:
            name = metadata["document"]
            verified_sources[name] = verified_sources.get(name, 0) + 1
    if verified_sources != {name: stats["chunks"] for name, stats in source_stats.items()}:
        raise RuntimeError("Chroma coverage differs from the extracted corpus")


def ingest_docs(data_path=None, persist_dir=None, *, force=False, model=EMBEDDING_MODEL) -> dict:
    data_path = project_path(data_path or os.getenv("DATA_PATH") or "data")
    root = project_path(persist_dir) if persist_dir is not None else index_root()
    files, profiles, inventory = corpus_inventory(data_path)
    fingerprint = hashlib.sha256(json.dumps(
        [inventory, profiles, INDEX_SCHEMA, "provisions-v3", EMBEDDING_REVISIONS[model],
         CHUNK_SIZE, CHUNK_OVERLAP],
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    active = None
    previous = None
    try:
        active = read_index_manifest(root, model)
        previous = validate_collection(root / "snapshots" / active["snapshot"], active["chunks"], model)
        if not force and active.get("fingerprint") == fingerprint:
            logger.info("Reusing the validated index for the same corpus")
            return active
    except IndexUnavailable:
        if (root / ACTIVE_INDEX_FILE).is_file() and not active:
            try:
                incompatible = json.loads((root / ACTIVE_INDEX_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                incompatible = {}
            if isinstance(incompatible, dict) and incompatible.get("embedding_model") in EMBEDDING_REVISIONS and incompatible["embedding_model"] != model:
                raise IndexUnavailable("Use a separate index directory for a different embedding model.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents, source_stats, failures = [], {}, []
    for path in files:
        try:
            chunks, stats = extract_chunks(path, profiles.get(path.name, {}), inventory[path.name], splitter)
        except Exception as error:
            failures.append(f"{path.name}: {error}")
            continue
        documents.extend(chunks)
        source_stats[path.name] = stats
        logger.info("%s: %s pages, %s empty, %s chunks",
                    path.name, stats["pages"], len(stats["empty_pages"]), stats["chunks"])
    if failures:
        raise RuntimeError("Index was not published because PDF extraction failed:\n" + "\n".join(failures))
    if corpus_inventory(data_path)[1:] != (profiles, inventory):
        raise RuntimeError("Corpus changed during extraction. Run ingestion again.")

    cached = {}
    if previous is not None and not force:
        old_collection = previous.get_collection(COLLECTION_NAME, embedding_function=None)
        for offset in range(0, active["chunks"], 512):
            rows = old_collection.get(offset=offset, limit=512, include=["documents", "embeddings"])
            cached.update(zip(rows["documents"], rows["embeddings"]))
    snapshot = uuid4().hex
    path = root / "snapshots" / snapshot
    path.mkdir(parents=True)
    client = connect_snapshot(path)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={
            "embedding_revision": EMBEDDING_REVISIONS[model],
            "index_schema": INDEX_SCHEMA,
        },
        embedding_function=None,
    )
    generated = 0
    for offset in range(0, len(documents), INDEX_BATCH):
        batch = documents[offset:offset + INDEX_BATCH]
        texts = [doc.page_content for doc in batch]
        missing = list(dict.fromkeys(text for text in texts if text not in cached))
        if missing:
            embedded = get_embeddings(model).embed_documents(missing)
            if len(embedded) != len(missing):
                raise RuntimeError("Embedding count does not match the requested batch")
            cached.update(zip(missing, embedded))
            generated += len(missing)
        vectors = [list(map(float, cached[text])) for text in texts]
        if len(vectors) != len(batch) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise RuntimeError("Embedding dimensions do not match the configured model")
        collection.add(
            ids=[doc.id for doc in batch],
            documents=texts,
            metadatas=[doc.metadata for doc in batch],
            embeddings=vectors,
        )
        if offset % (INDEX_BATCH * 20) == 0:
            logger.info("Indexed %s/%s chunks", offset + len(batch), len(documents))
    validate_snapshot(collection, documents, source_stats)
    probe = collection.query(query_embeddings=[vectors[-1]], n_results=1, include=["metadatas"])
    if not probe["ids"] or not probe["ids"][0]:
        raise RuntimeError("Chroma did not return a result for the validation vector")
    manifest = {
        "schema": INDEX_SCHEMA,
        "snapshot": snapshot,
        "collection": COLLECTION_NAME,
        "embedding_model": model,
        "embedding_revision": EMBEDDING_REVISIONS[model],
        "dimension": EMBEDDING_DIMENSION,
        "fingerprint": fingerprint,
        "chunks": len(documents),
        "new_vectors": generated,
        "sources": source_stats,
    }
    if corpus_inventory(data_path)[1:] != (profiles, inventory):
        raise RuntimeError("Corpus changed during indexing. Run ingestion again.")
    pending = root / f".{snapshot}.json"
    try:
        serialized = json.dumps(manifest, indent=2, ensure_ascii=False)
        (path / "manifest.json").write_text(serialized, encoding="utf-8")
        if active:
            backup = root / "snapshots" / active["snapshot"] / "manifest.json"
            if backup.parent.is_dir() and not backup.exists():
                backup.write_text(json.dumps(active, indent=2, ensure_ascii=False), encoding="utf-8")
        pending.write_text(serialized, encoding="utf-8")
        pending.replace(root / ACTIVE_INDEX_FILE)
    finally:
        pending.unlink(missing_ok=True)
    return manifest


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="Directory containing PDF files and optional corpus.json")
    parser.add_argument("--persist-dir", help="Directory for Chroma snapshots and the index manifest")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the corpus is unchanged")
    parser.add_argument("--embedding-model", choices=EMBEDDING_REVISIONS, default=EMBEDDING_MODEL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        manifest = ingest_docs(args.data_path, args.persist_dir, force=args.force, model=args.embedding_model)
    except Exception as error:
        logger.error("%s", error)
        return 1
    logger.info("Active index: %s PDFs, %s chunks", len(manifest["sources"]), manifest["chunks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
