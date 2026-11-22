import json
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.db import IndexUnavailable
from backend.hosted import GeminiAuthenticationFailed, RateLimited
from backend.main import create_app
from backend.rag_pipeline import GenerationTimeout, LegalAssistant, ModelUnavailable, Ollama, ProviderUnavailable, validate_claims
from backend.retrieval import LegalRetriever, legal_context
from config import Settings


def document(id_, content, act, section, **extra):
    return Document(id=id_, page_content=content, metadata={
        "document": f"{act}.pdf", "act": act, "section": section,
        "pdf_page": 7, "page_label": "7", "authority_type": "primary_legislation",
        "jurisdiction": "IN", "legal_status": "current", **extra,
    })


RTI = document("rti", "Where the information sought concerns the life or liberty of a person, "
                      "the information shall be provided within forty-eight hours of receipt.",
               "Right to Information Act", "7", document_order=4)
OLD = document("old", "Information about a cognizable offence shall be recorded by the officer.",
               "Code of Criminal Procedure", "154", act_id="CrPC", legal_status="legacy_subject_to_savings")
NEW = document("new", "Every information relating to the commission of a cognizable offence "
                      "may be given orally to the police officer or as an electronic communication.",
               "Bharatiya Nagarik Suraksha Sanhita", "173", act_id="BNSS",
               effective_from="2024-07-01", document_order=4)
ADJACENT = document("adjacent", "A person aggrieved by a refusal may send the information "
                               "in writing and by post to the Superintendent of Police.",
                    "Bharatiya Nagarik Suraksha Sanhita", "173", act_id="BNSS", pdf_page=8,
                    effective_from="2024-07-01", document_order=5)


class Store:
    def __init__(self, docs):
        self.docs = docs

    def get(self, *, limit, offset, include):
        batch = self.docs[offset:offset + limit]
        return {
            "ids": [doc.id for doc in batch],
            "documents": [doc.page_content for doc in batch],
            "metadatas": [doc.metadata for doc in batch],
        }

    def similarity_search(self, query, k):
        return self.docs[:k]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = LegalRetriever(Store([OLD, NEW, ADJACENT, RTI]))

    def test_exact_reference_and_adjacent_passage(self):
        evidence = self.retriever.retrieve("What if police refuse under BNSS section 173?")
        self.assertEqual(evidence.status, "relevant")
        self.assertEqual({doc.id for doc in evidence.documents}, {"new", "adjacent"})

    def test_applicability_and_scope(self):
        modern = self.retriever.retrieve("In 2025 what can I do if police refuse to record a cognizable offence?")
        self.assertEqual(modern.documents[0].id, "adjacent")
        self.assertNotIn("old", {doc.id for doc in modern.documents})
        historical = self.retriever.retrieve("Under the old CrPC what section governed FIRs?")
        self.assertEqual([doc.id for doc in historical.documents], ["old"])
        self.assertEqual(legal_context("My criminal investigation", []).status, "insufficient")
        self.assertEqual(self.retriever.retrieve("What is the weather in Pune?").status, "out_of_scope")
        self.assertEqual(self.retriever.retrieve("What is underwater ceramic permeability?").status, "insufficient")

    def test_follow_up_history_is_request_scoped(self):
        question = "What if someone's life is at risk?"
        context = ["What is the RTI response deadline?"]
        resolved = self.retriever.retrieve(question, context)
        self.assertEqual(resolved.documents[0].id, "rti")
        self.assertIn("right to information", resolved.context.scenario)
        self.assertEqual(self.retriever.retrieve(question).context.scenario, question)

    def test_reranker_called_once_and_cannot_override_exact_reference(self):
        scorer = Mock()
        scorer.predict.side_effect = lambda pairs, **_: list(range(len(pairs)))
        retrieval = LegalRetriever(Store([OLD, RTI, NEW, ADJACENT]), reranker=scorer)
        evidence = retrieval.retrieve("BNSS section 173")
        scorer.predict.assert_called_once()
        self.assertTrue(all(doc.metadata["section"] == "173" for doc in evidence.documents))

    def test_consumer_filing_limit_excludes_distinct_procedural_deadlines(self):
        wrong = document("admission", "The consumer complaint shall ordinarily be admitted within "
                         "twenty-one days from filing the consumer complaint.",
                         "Consumer Protection Act", "36")
        right = document("limitation", "The Commission shall not admit a complaint unless it is filed within "
                         "two years from the cause of action. A delay can be excused for sufficient cause.",
                         "Consumer Protection Act", "69")
        retrieval = LegalRetriever(Store([wrong, right]))
        result = retrieval.retrieve("What is the time limit for a consumer complaint, and can delay be excused?")
        self.assertEqual([item.id for item in result.documents], ["limitation"])

    def test_fir_refusal_keeps_recording_and_escalation_in_one_verified_passage(self):
        opening = document("start", "(4) Any person aggrieved by a refusal of the officer to record the information "
                           "may send the substance of such information in writing to the Superintendent of Police concerned",
                           "Bharatiya Nagarik Suraksha Sanhita", "173", act_id="BNSS", document_order=1)
        end = document("end", "the Superintendent of Police concerned who may direct investigation, "
                       "failing which the person may make an application to the Magistrate.",
                       "Bharatiya Nagarik Suraksha Sanhita", "173", act_id="BNSS", document_order=2)
        result = LegalRetriever(Store([opening, end])).retrieve(
            "What if police refuse to record a cognizable offence under BNSS section 173?"
        )
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].metadata["subsection"], "4")
        self.assertIn("Superintendent of Police", result.documents[0].page_content)
        self.assertIn("application to the Magistrate", result.documents[0].page_content)


class GroundingTests(unittest.TestCase):
    def test_general_fallback_has_no_citations_and_retains_followup_questions(self):
        provider = Mock()
        provider.generate_general.return_value = (json.dumps({
            "text": "Registration can depend on turnover, location, and the type of supplies. Check current official guidance.",
            "outside_legal": True,
        }), "simulated")
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([])))
        question = "What are the GST registration thresholds?"
        result = assistant.answer(question, history=["How does GST apply to businesses?"], allow_general=True)
        self.assertEqual(result["answer_kind"], "general")
        self.assertFalse(result["citations"])
        self.assertFalse(result["general_information"]["outside_legal"])
        request = json.loads(provider.generate_general.call_args.args[0][1]["content"])
        self.assertEqual(request["recent_questions"], ["How does GST apply to businesses?"])
        provider.generate.assert_not_called()
        provider.generate_with_general.assert_not_called()

    def test_legal_general_output_rejects_specific_rules_and_fabricated_sources(self):
        provider = Mock()
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([])))
        for text in ("Under the invented Tax Compliance Act, registration applies.",
                     "Every seller must register.", "The registration rule currently applies to every seller.",
                     "The threshold is twenty lakh rupees.",
                     "The threshold is set by Section 999.", "Check the rules here [S9]."):
            with self.subTest(text=text):
                provider.generate_general.return_value = (json.dumps({"text": text, "outside_legal": True}), "simulated")
                result = assistant.answer("What are the GST registration thresholds?", allow_general=True)
                self.assertNotIn("general_information", result)
                self.assertFalse(result["citations"])

    def test_partial_answer_keeps_general_information_out_of_verified_claims(self):
        provider = Mock()
        hma = document("hma", "The bridegroom has completed twenty-one years and the bride has completed eighteen years.",
                       "Hindu Marriage Act", "5")
        claims = [{"source": "S1", "text": hma.page_content, "quote": hma.page_content}]
        general = "Keeping identity documents together can make it easier to check local procedures."
        provider.generate_with_general.return_value = (json.dumps({
            "claims": claims, "incomplete": True, "general": {"text": general, "outside_legal": False},
        }), "simulated")
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([hma])))
        result = assistant.answer("What minimum ages does the Hindu Marriage Act prescribe and what paperwork may help?",
                                  allow_general=True)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["general_information"]["text"], general)
        self.assertNotIn(general, result["answer"])
        self.assertEqual(result["citations"][0]["section"], "5")
        self.assertNotIn(general, result["citations"][0]["excerpt"])
        provider.generate_with_general.assert_called_once()
        provider.generate_general.assert_not_called()

    def test_missing_provision_and_verified_only_skip_fallback(self):
        provider = Mock()
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([NEW])))
        for question, allowed in (("BNS Section 999", True), ("BNS 999", True),
                                  ("My criminal investigation", True), ("What are GST thresholds?", False)):
            with self.subTest(question=question):
                result = assistant.answer(question, allow_general=allowed)
                self.assertEqual(result["answer_kind"], "abstention")
                self.assertFalse(result["citations"])
        provider.generate.assert_not_called()
        provider.generate_with_general.assert_not_called()
        provider.generate_general.assert_not_called()

    def test_retrieval_and_provider_failures_stay_failures(self):
        provider = Mock()
        unavailable = Mock(side_effect=IndexUnavailable("missing"))
        with self.assertRaises(IndexUnavailable):
            LegalAssistant(Settings(), provider=provider, retriever_factory=unavailable).answer(
                "What are the GST registration thresholds?", allow_general=True,
            )
        provider.generate_general.assert_not_called()
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([])))
        for error in (ProviderUnavailable, GeminiAuthenticationFailed, GenerationTimeout, RateLimited):
            with self.subTest(error=error):
                provider.generate_general.side_effect = error("unavailable")
                with self.assertRaises(error):
                    assistant.answer("What are the GST registration thresholds?", allow_general=True)

    def test_reject_fabricated_ids_quotes_and_numbers(self):
        sources = {"S1": {"excerpt": RTI.page_content, "section": "7",
                          "act": "Right to Information Act"}}
        valid = {"source": "S1", "quote": RTI.page_content,
                 "text": "Information concerning life or liberty must be provided within 48 hours."}
        for invalid in ({**valid, "source": "S9"}, {**valid, "quote": "No deadline applies to this request."},
                        {**valid, "text": valid["text"].replace("48", "96")},
                        {**valid, "text": "Section 44 requires information about life or liberty."},
                        {**valid, "text": "Is information concerning life or liberty due in 48 hours?"},
                        {**valid, "text": "If a public officer fails to decide, information must be provided within 48 hours."}):
            accepted, incomplete = validate_claims(json.dumps({"claims": [valid, invalid], "incomplete": False}), sources)
            self.assertEqual(len(accepted), 1)
            self.assertTrue(incomplete)

    def test_model_is_skipped_for_out_of_scope(self):
        provider = Mock()
        assistant = LegalAssistant(Settings(), provider=provider)
        self.assertEqual(assistant.answer("What is the weather in Pune?")["status"], "out_of_scope")
        provider.generate.assert_not_called()
        self.assertIsNone(assistant.retriever)

    def test_invalid_generation_can_return_labeled_verbatim_excerpt(self):
        provider = Mock()
        provider.generate.return_value = ('{"claims":[{"source":"S99"}]}', "local-test")
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([RTI])))
        answer = assistant.answer("When must an RTI request concerning life or liberty be answered?")
        self.assertEqual(answer["status"], "partial")
        self.assertEqual(answer["answer_kind"], "excerpt")
        self.assertEqual(answer["citations"][0]["quotes"], [RTI.page_content])
        self.assertNotIn("S99", answer["answer"])
        exact = assistant.answer("What does RTI section 7 say?")
        self.assertEqual(exact["answer_kind"], "excerpt")
        self.assertEqual(exact["citations"][0]["section"], "7")

    def test_unambiguous_section_lookup_uses_verified_heading(self):
        source = document("theft", "66C. Punishment for identity theft. Dishonest use of an electronic signature.",
                          "Information Technology Act", "66C", section_heading="Punishment for identity theft")
        provider = Mock()
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([source])))
        answer = assistant.answer("Which IT Act section covers identity theft?")
        self.assertEqual(answer["answer_kind"], "metadata")
        self.assertIn("Section 66C", answer["answer"])
        self.assertEqual(answer["citations"][0]["section"], "66C")
        provider.generate.assert_not_called()

    def test_rti_deadlines_use_exact_matching_statutory_language(self):
        section = document("rti-deadlines", "Disposal of request. The officer shall, as expeditiously "
                           "as possible, and in any case within thirty days of the receipt of the request, "
                           "provide the information. Provided that where the information sought for concerns "
                           "the life or liberty of a person, the same shall be provided within forty-eight hours "
                           "of the receipt of the request.",
                           "Right to Information Act", "7")
        provider = Mock()
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([section])))
        question = "What is the standard RTI response deadline?"
        normal = assistant.answer(question)
        urgent = assistant.answer("What if my life is at risk?", history=[question])
        self.assertIn("thirty days", normal["answer"])
        self.assertNotIn("forty-eight hours", normal["answer"])
        self.assertIn("forty-eight hours", urgent["answer"])
        self.assertEqual(urgent["citations"][0]["pdf_page"], 7)
        provider.generate.assert_not_called()

    def test_local_model_failure_can_fall_back_but_timeout_is_not_retried(self):
        provider = Ollama(Settings(model="qwen2.5:7b-instruct-q4_K_M"))
        try:
            with patch.object(provider, "local", side_effect=[ModelUnavailable(), ("{}", "fallback")]) as call:
                self.assertEqual(provider.generate([]), ("{}", "fallback"))
                self.assertEqual(call.call_count, 2)
            with patch.object(provider, "local", side_effect=GenerationTimeout()) as call:
                with self.assertRaises(GenerationTimeout):
                    provider.generate([])
                call.assert_called_once()
        finally:
            provider.close()


class APITests(unittest.TestCase):
    def test_compatible_routes_preserve_complete_answer_and_citations(self):
        provider = Mock()
        provider.generate.return_value = (json.dumps({
            "claims": [{"text": "Information concerning life or liberty must be provided within 48 hours.",
                        "source": "S1", "quote": RTI.page_content}],
            "incomplete": False,
        }), "local-test")
        assistant = LegalAssistant(Settings(), provider=provider, retriever=LegalRetriever(Store([RTI])))
        with TestClient(create_app(Settings(), assistant=assistant)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            provider.generate.assert_not_called()
            for route in ("/chat", "/query/"):
                response = client.post(route, json={"question": "What if someone's life is at risk?",
                                                    "history": ["What is the RTI deadline?"]})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("48 hours. [S1]", data["answer"])
                self.assertEqual(data["response"], data["answer"])
                self.assertEqual(data["citations"][0]["pdf_page"], 7)

    def test_service_errors_are_typed(self):
        service = Mock()
        service.answer.side_effect = [ProviderUnavailable("private detail"), GenerationTimeout()]
        with TestClient(create_app(Settings(), assistant=service)) as client:
            for status, code in [(503, "unavailable"), (504, "timeout")]:
                response = client.post("/query/", json={"question": "What is the RTI limit?"})
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], code)
                self.assertNotIn("private detail", response.text)

    def test_readiness_checks_availability_without_generation(self):
        service = Mock()
        service.provider.ready.side_effect = [False, True]
        with patch("backend.main.check_index"), TestClient(create_app(Settings(), assistant=service)) as client:
            unavailable = client.get("/ready")
            self.assertEqual(unavailable.status_code, 503)
            self.assertEqual(unavailable.json()["component"], "generation")
            self.assertEqual(client.get("/ready").json(), {"status": "ready"})
            response = client.post("/query/", json={"question": "private " * 250})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("private ", response.text)
        service.answer.assert_not_called()
