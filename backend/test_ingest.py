import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.db import EMBEDDING_DIMENSION, PROJECT_ROOT, IndexUnavailable, get_vector_store
from backend.ingest import ProvisionTracker, ingest_docs


class FixedEmbeddings:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


class IngestionTests(unittest.TestCase):
    def test_index_round_trip_and_failed_rebuild_preserves_active_snapshot(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            pdf = next((PROJECT_ROOT / "data").glob("RTI*.pdf"))
            shutil.copyfile(pdf, data / pdf.name)
            store = root / "chroma"

            with self.assertRaises(IndexUnavailable):
                get_vector_store(store)
            self.assertFalse(store.exists())

            embeddings = FixedEmbeddings()
            with (
                patch("backend.ingest.get_embeddings", return_value=embeddings),
                patch("backend.db.get_embeddings", return_value=embeddings),
            ):
                first = ingest_docs(data, store)
                second = ingest_docs(data, store)
                self.assertEqual(first, second)
                db = get_vector_store(store)
                stored = db.get(where={"pdf_page": 7}, include=["documents", "metadatas"])
                rti = next(
                    metadata for text, metadata in zip(stored["documents"], stored["metadatas"])
                    if "forty-eight hours" in text
                )
                self.assertEqual(rti["document"], pdf.name)
                self.assertEqual(rti["page"], 6)
                self.assertEqual(rti["page_label"], "7")
                self.assertEqual(len(db.get()["ids"]), first["chunks"])

                profile = data / "corpus.json"
                profile.write_text(json.dumps({pdf.name: {"authority_type": "primary_legislation"}}),
                                   encoding="utf-8")
                metadata_update = ingest_docs(data, store)
                self.assertEqual(metadata_update["new_vectors"], 0)
                self.assertNotEqual(first["snapshot"], metadata_update["snapshot"])
                db = get_vector_store(store)
                self.assertTrue(all(row["authority_type"] == "primary_legislation"
                                    for row in db.get(include=["metadatas"])["metadatas"]))

                (data / "broken.pdf").write_bytes(b"invalid PDF")
                with self.assertRaisesRegex(RuntimeError, "broken.pdf"):
                    ingest_docs(data, store)
                self.assertEqual(get_vector_store(store).get()["ids"], db.get()["ids"])

                (data / "broken.pdf").unlink()
                profile.unlink()
                (data / pdf.name).unlink()
                replacement = "The Hindu marriage act.pdf"
                shutil.copyfile(PROJECT_ROOT / "data" / replacement, data / replacement)
                rebuilt = ingest_docs(data, store)
                self.assertNotEqual(first["snapshot"], rebuilt["snapshot"])
                current = get_vector_store(store).get(include=["metadatas"])
                self.assertEqual({row["document"] for row in current["metadatas"]}, {replacement})

    def test_bns_commencement_exception_tracks_continuation_across_pages(self):
        tracker = ProvisionTracker({"act_id": "BNS", "gazette_sections": 107})
        tracker.last_gazette_section = 105
        first = list(tracker.segments(
            "106. (1) General provision.\n(2) Whoever causes death in the specified circumstances.",
            34,
        ))
        continued = list(tracker.segments("Continuation.\n107. A different section.", 35))
        self.assertEqual(first[-1][1]["subsection"], "2")
        self.assertEqual(continued[0][1]["legal_status"], "excluded_at_commencement")
        self.assertEqual(continued[-1][1], {"section": "107"})
        tracker.validate()

    def test_bnss_schedule_exception_does_not_inherit_last_section(self):
        tracker = ProvisionTracker({"act_id": "BNSS", "gazette_sections": 1})
        segments = list(tracker.segments(
            "1. (1) Act text.\nTHE FIRST SCHEDULE\n106(2) Causing death by rash and negligent driving.",
            1,
        ))
        metadata = segments[-1][1]
        self.assertEqual(metadata["authority_type"], "schedule")
        self.assertEqual(metadata["legal_status"], "mixed_commencement")
        self.assertNotIn("section", metadata)
        tracker.validate()

    def test_constitution_fourth_schedule_does_not_inherit_article_395(self):
        from pypdf import PdfReader

        data = PROJECT_ROOT / "data"
        name = "constitution of india.pdf"
        manifest = json.loads((data / "corpus.json").read_text(encoding="utf-8"))
        tracker = ProvisionTracker(manifest[name])
        pdf = PdfReader(data / name)
        body = list(tracker.segments(pdf.pages[282].extract_text(), 283))
        self.assertTrue(any(metadata.get("section") == "395" for _, metadata in body))
        schedule = list(tracker.segments(pdf.pages[304].extract_text(), 305))
        self.assertIn("Fourth Schedule", schedule[0][0])
        self.assertEqual(schedule[0][1], {"authority_type": "schedule"})

    def test_article_15_wrapped_heading_ends_article_14(self):
        from pypdf import PdfReader

        tracker = ProvisionTracker({"provision_type": "article", "body_page": 33, "last_body_page": 283})
        page = PdfReader(PROJECT_ROOT / "data" / "constitution of india.pdf").pages[36].extract_text()
        article_14 = next(text for text, meta in tracker.segments(page, 37) if meta.get("section") == "14")
        self.assertNotIn("Prohibition of discrimination", article_14)
        self.assertNotIn("No citizen shall", article_14)
