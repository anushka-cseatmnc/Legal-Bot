import json
import unittest
from datetime import date
from functools import lru_cache
from unittest.mock import Mock

from langchain_core.documents import Document
from pypdf import PdfReader

from backend.db import PROJECT_ROOT, IndexUnavailable
from backend.hosted import RateLimited
from backend.transition import applicability, generate_explanation, hybrid_evidence, load_catalog, search, textual_diff


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def lookup(self, pair, query):
        return [mapping.id for mapping in search(self.catalog, pair, query)]

    def test_act_section_subsection_and_topic_search(self):
        self.assertEqual(self.lookup("BNS", "section 34 of IPC"), ["common-intention"])
        self.assertEqual(self.lookup("BNS", "S. 3(5) BNS"), ["common-intention"])
        self.assertEqual(self.lookup("BNSS", "zero FIR"), ["fir"])
        self.assertEqual(self.lookup("BSA", "Indian Evidence Act, 1872 section 65b"), ["electronic-records"])
        self.assertIn("digital-status", self.lookup("BSA", "Bharatiya Sakshya Adhiniyam, 2023"))
        self.assertEqual(self.lookup("BSA", "BSA 61"), ["digital-status"])
        self.assertEqual(self.lookup("BSA", "IEA 22A"), ["oral-admissions"])

    def test_no_result_for_wrong_regime_or_nonexistent_section(self):
        for pair, query in (("BNS", "IPC 999"), ("BNS", "IPC 303"), ("BSA", "IPC 34"),
                            ("BNS", "BNS 106(3)"), ("BNSS", "CrPC 173"),
                            ("BSA", "BSA 65B"), ("BSA", "weather")):
            with self.subTest(query=query):
                self.assertEqual(self.lookup(pair, query), [])

    def test_split_merge_and_unmatched_provisions(self):
        mappings, passages = self.catalog.mappings, self.catalog.passages
        self.assertEqual(self.lookup("BNS", "IPC 376"), ["rape-punishment"])
        self.assertEqual({passages[key].section for key in mappings["rape-punishment"].new}, {"64", "65"})
        self.assertEqual(self.lookup("BNSS", "BNSS 35(5)"), ["notice"])
        self.assertEqual(self.lookup("BNS", "BNS 303(2)"), ["theft"])
        self.assertEqual(len(mappings["theft"].old), 2)
        self.assertFalse(mappings["oral-admissions"].new)
        self.assertFalse(mappings["digital-status"].old)

    def test_each_verified_excerpt_matches_the_exact_pdf_page(self):
        @lru_cache(maxsize=None)
        def page(filename, number):
            return " ".join(reader(filename).pages[number - 1].extract_text().split())

        @lru_cache(maxsize=None)
        def reader(filename):
            return PdfReader(PROJECT_ROOT / "data" / filename)

        for key, passage in self.catalog.passages.items():
            with self.subTest(source=key):
                filename = self.catalog.acts[passage.act_id]["document"]
                self.assertIn(passage.text, page(filename, passage.page))

    def test_historical_saved_proceedings_and_uncertainty(self):
        mappings = self.catalog.mappings
        before, after = date(2024, 6, 30), date(2024, 7, 1)
        self.assertEqual(applicability(self.catalog, mappings["theft"], before, "not_pending", "india").status, "legacy")
        self.assertEqual(applicability(self.catalog, mappings["theft"], after, "pending", "india").status, "successor")
        self.assertEqual(applicability(self.catalog, mappings["fir"], after, "unsure", "india").status, "clarify")
        self.assertIn("BNSS-531-next", applicability(self.catalog, mappings["fir"], after, "pending", "india").sources)
        evidence = applicability(self.catalog, mappings["electronic-records"], after, "pending", "india")
        self.assertEqual((evidence.status, evidence.sources), ("legacy", ("BSA-170",)))
        self.assertEqual(applicability(self.catalog, mappings["fir"], before, "not_pending", "india").status, "conditional")
        self.assertEqual(applicability(self.catalog, mappings["fir"], after, "not_pending", "india").status, "successor")

    def test_exceptions_and_jurisdiction(self):
        mappings = self.catalog.mappings
        after = date(2025, 1, 1)
        result = applicability(self.catalog, mappings["non-reporting"], after, None, "india")
        self.assertEqual(result.status, "exception")
        self.assertIn("S.O. 848(E)", result.text)
        self.assertIn("do not establish a later commencement", result.text)
        result = applicability(self.catalog, mappings["fir"], after, "not_pending", "special")
        self.assertEqual(result.status, "exception")
        self.assertIn("BNSS-1", result.sources)
        self.assertEqual(applicability(self.catalog, mappings["fir"], after, "pending", "outside").status, "outside")
        self.assertEqual(applicability(self.catalog, mappings["fir"], after, "pending", "unsure").status, "clarify")

    def test_only_matching_retrieved_pages_and_excerpts_are_prioritized(self):
        entry = self.catalog.mappings["electronic-records"]
        passage = self.catalog.passages["BSA-63-4"]
        wrong = Document(id="wrong", page_content=passage.text, metadata={
            "act_id": "BSA", "section": "63", "document": "BSA 2023.pdf", "pdf_page": 22,
        })
        right = Document(id="right", page_content=passage.text, metadata={**wrong.metadata, "pdf_page": 23})
        retriever = Mock(documents={"right": right})
        retriever.retrieve.return_value = Mock(documents=[wrong, right])
        retriever.rank_supplemental.side_effect = lambda _, docs: list(reversed(docs))
        evidence = hybrid_evidence(self.catalog, entry, retriever)
        self.assertEqual(list(evidence), ["IEA-65B-4", "IEA-65B-1", "BSA-63-4", "BSA-63-1"])
        self.assertEqual(evidence["BSA-63-4"]["pdf_page"], 23)
        self.assertEqual(evidence["IEA-65B-4"]["pdf_page"], 35)

    def test_citations_must_be_present_on_both_sides(self):
        mapping = self.catalog.mappings["common-intention"]
        legacy, successor = (self.catalog.passages[key] for key in (*mapping.old, *mapping.new))
        claims = [{"text": passage.text, "quote": passage.text, "source": passage.id}
                  for passage in (legacy, successor)]
        provider = Mock()
        provider.generate.return_value = json.dumps({"claims": claims, "incomplete": False}), "gemini"
        missing_index = Mock(side_effect=IndexUnavailable("missing"))
        notes, message = generate_explanation(self.catalog, mapping, missing_index, provider)
        self.assertEqual({note[1] for note in notes}, {legacy.id, successor.id})
        self.assertIsNone(message)
        claims[1]["quote"] = "This citation is fabricated and should be rejected."
        provider.generate.return_value = json.dumps({"claims": claims, "incomplete": False}), "gemini"
        notes, message = generate_explanation(self.catalog, mapping, missing_index, provider)
        self.assertFalse(notes)
        self.assertIn("both sides", message)

    def test_missing_side_and_hosted_rate_limit(self):
        provider = Mock()
        retriever = Mock(side_effect=IndexUnavailable("missing"))
        notes, message = generate_explanation(self.catalog, self.catalog.mappings["oral-admissions"], retriever, provider)
        self.assertFalse(notes)
        self.assertIn("both sides", message)
        provider.generate.assert_not_called()
        provider.generate.side_effect = RateLimited("quota")
        with self.assertRaises(RateLimited):
            generate_explanation(self.catalog, self.catalog.mappings["fir"], retriever, provider)

    def test_textual_diff(self):
        self.assertEqual(textual_diff("a b", "a c"), [("equal", "a", "a"), ("replace", "b", "c")])
