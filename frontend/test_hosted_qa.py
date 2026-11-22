"""Full app, retrieval, SDK, and citation tests. Only the live class calls Google."""

import json
import os
import re
import unittest
from time import sleep
from unittest.mock import patch

import httpx
import streamlit as st
from google import genai
from google.genai import types
from streamlit.testing.v1 import AppTest

from backend.cloud_service import DemoService
from backend.db import PROJECT_ROOT, project_path
from backend.index_bundle import LOCAL_BUNDLE
from backend.hosted import Gemini

THEFT = "What is the punishment for theft under BNS Section 303?"
FOLLOW_UP = "What about a second conviction?"
MARRIAGE = "What minimum ages does the Hindu Marriage Act prescribe?"
GST = "What are the GST registration thresholds in India?"
MACHINE_LEARNING = "What is the difference between supervised and reinforcement learning?"
HAS_INDEX = (
    (os.getenv("LEGALBOT_INDEX_SHA256") and os.getenv("LEGALBOT_INDEX_BUNDLE"))
    or (project_path(LOCAL_BUNDLE).is_file() and project_path(LOCAL_BUNDLE).with_suffix(".sha256").is_file())
)


class HostedQAChecks:
    def check_chat(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        for index, (question, document, section, page, expected) in enumerate((
            (THEFT, "BNS 2023.pdf", "303", 79, (r"(?:three|3) years", r"fine")),
            (FOLLOW_UP, "BNS 2023.pdf", "303", 79, (
                r"\b(?:one|1)\b\s*(?:year|to\b|and\b|[-–])", r"(?:five|5)\s*[- ]?\s*years", r"fine",
            )),
            (MARRIAGE, "The Hindu marriage act.pdf", "5", 5, (r"(?:twenty[- ]one|21)", r"(?:eighteen|18)")),
        )):
            if index:
                sleep(2.1)
            app.chat_input[0].set_value(question).run()
            self.assertFalse(app.exception)
            result = app.session_state.turns[-1][1]
            self.assertEqual(result.get("answer_kind"), "generated",
                             f"Turn {index + 1} did not produce a citation-validated Gemini answer.")
            for term in expected:
                self.assertRegex(result["answer"].lower(), term)
            cited = [source for source in result["citations"] if (
                source["document"], source["section"], source["pdf_page"]
            ) == (document, section, page)]
            self.assertTrue(cited)
            for source in cited:
                self.assertTrue(source["quotes"])
                for quote in source["quotes"]:
                    self.assertIn(" ".join(quote.split()), " ".join(source["excerpt"].split()))
                self.assertTrue(any(f"[{source['id']}]" in card.label and f"PDF p. {page}" in card.label
                                    for card in app.expander))
            self.assertTrue(any(result["answer"] == element.value for element in app.markdown))
            self.assertNotIn("fabricated", result["answer"])
            if question == FOLLOW_UP:
                self.assertIn(THEFT, result["legal_context"]["query"])


@unittest.skipUnless(HAS_INDEX, "Provide the verified index bundle.")
class SimulatedGeminiQATest(HostedQAChecks, unittest.TestCase):
    def test_general_modes_verified_only_and_provider_error(self):
        requests = []

        def transport(request):
            body = json.loads(request.content)
            self.assertEqual(body["generationConfig"]["responseJsonSchema"]["required"], ["text", "outside_legal"])
            question = json.loads(body["contents"][0]["parts"][0]["text"])["question"]
            requests.append(question)
            if len(requests) > 2:
                return httpx.Response(503, json={"error": {"code": 503, "message": "simulated upstream outage"}})
            response = {
                "text": ("Registration can depend on turnover and the type of supplies." if question == GST
                         else "Supervised learning learns from labeled examples; reinforcement learning learns from rewards."),
                "outside_legal": question == MACHINE_LEARNING,
            }
            return httpx.Response(200, json={"candidates": [{"content": {
                "parts": [{"text": json.dumps(response)}], "role": "model",
            }, "finishReason": "STOP"}]})

        def ask(app, question):
            if app.session_state.turns:
                sleep(2.05)
            app.chat_input[0].set_value(question).run()
            self.assertFalse(app.exception)
            return app.session_state.turns[-1][1]

        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(
            api_version="v1beta", httpx_client=httpx.Client(transport=httpx.MockTransport(transport)),
        )) as client, patch("frontend.app.DemoService", side_effect=lambda config: DemoService(
            config, provider=Gemini(config.provider, client=client, pause=lambda _: None),
        )):
            app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=120).run()
            for question, outside in ((GST, False), (MACHINE_LEARNING, True)):
                result = ask(app, question)
                self.assertEqual(result["answer_kind"], "general")
                self.assertEqual(result["general_information"]["outside_legal"], outside)
                self.assertFalse(result["citations"])
            self.assertTrue(any("Outside LegalBot’s legal specialization" in item.value for item in app.caption))
            result = ask(app, "What does BNS Section 999 say?")
            self.assertEqual(result["answer_kind"], "abstention")
            self.assertFalse(result["citations"])
            self.assertNotIn("general_information", result)
            app.toggle[0].set_value(True).run()
            result = ask(app, GST)
            self.assertNotIn("general_information", result)
            app.toggle[0].set_value(False).run()
            result = ask(app, GST)
            self.assertEqual(result["status"], "error")
            self.assertTrue(any("temporarily unavailable" in error.value for error in app.error))
        self.assertEqual(requests, [GST, MACHINE_LEARNING, GST, GST])

    def test_real_retrieval_and_ui_with_simulated_gemini_http(self):
        calls = []

        def transport(request):
            self.assertEqual(request.url.host, "generativelanguage.googleapis.com")
            self.assertTrue(request.url.path.endswith(":generateContent"))
            body = json.loads(request.content)
            self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
            self.assertIn("responseJsonSchema", body["generationConfig"])
            payload = json.loads(body["contents"][0]["parts"][0]["text"])
            question = payload["question"]
            calls.append(question)
            if question.startswith(THEFT):
                pattern = r"Whoever commits theft shall be punished with imprisonment.*?or with both"
                expected = ("303", "shall be punished with community service")
            elif question.startswith(FOLLOW_UP):
                self.assertIn(THEFT, question)
                pattern = r"in case of second or subsequent conviction.*?and with fine"
                expected = ("303", "shall be punished with community service")
            else:
                self.assertEqual(question, MARRIAGE)
                pattern = r"the bridegroom has completed the age.*?at the time of the marriage"
                expected = ("5", "the bridegroom has completed")
            claim = None
            for source in payload["evidence"]:
                text = " ".join(source["text"].split())
                if source["section"] == expected[0] and expected[1] in text:
                    if match := re.search(pattern, text):
                        claim = {"text": match[0].capitalize(), "quote": match[0], "source": source["id"]}
                        break
            self.assertIsNotNone(claim, f"Turn {len(calls)} is missing the expected statutory evidence.")
            forged = {"text": "A fabricated answer would impose imprisonment for ninety years.",
                      "quote": claim["quote"], "source": "S999"}
            wrong_quote = {"text": "A fabricated passage would impose imprisonment for ninety years.",
                           "quote": "This is a fabricated quotation.", "source": claim["source"]}
            response = json.dumps({"claims": [claim, forged, wrong_quote], "incomplete": False})
            return httpx.Response(200, json={"candidates": [{"content": {
                "parts": [{"text": response}], "role": "model",
            }, "finishReason": "STOP"}]})

        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(
            api_version="v1beta", httpx_client=httpx.Client(transport=httpx.MockTransport(transport)),
        )) as client, patch("frontend.app.DemoService", side_effect=lambda config: DemoService(
            config, provider=Gemini(config.provider, client=client),
        )):
            self.check_chat()
        self.assertEqual(len(calls), 3)


@unittest.skipUnless(os.getenv("LEGALBOT_VERIFY_LIVE_GEMINI") == "1", "Live Gemini check is opt-in.")
class LiveGeminiQATest(HostedQAChecks, unittest.TestCase):
    def test_streamlit_legal_qa_with_live_gemini(self):
        from frontend.app import service

        st.cache_resource.clear()
        self.assertTrue(service().config.provider.api_key,
                        "Set [legalbot].api_key in .streamlit/secrets.toml or set GEMINI_API_KEY.")
        self.check_chat()
