import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import httpx
import numpy as np
from google import genai
from google.genai import types

from backend.cloud_service import CloudSettings, DemoService
from backend.db import COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_REVISION, IndexUnavailable
from backend.hosted import (
    Busy, DEFAULT_MODEL, Gemini, GeminiAuthenticationFailed, GeminiModelUnavailable, GeminiNotConfigured,
    HostedSettings, RateLimited, RequestBudget,
)
from backend.index_bundle import LOCAL_BUNDLE, prepare_bundle, sha256
from backend.rag_pipeline import GenerationTimeout, ProviderUnavailable

MESSAGES = [{"role": "system", "content": "Use only indexed evidence."},
            {"role": "user", "content": '{"question":"Explain verified section."}'}]


class HostedTests(unittest.TestCase):
    def test_credential_resolution_and_safe_missing_key_diagnostic(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(CloudSettings.from_mapping({}).provider.api_key)
            with self.assertLogs("backend.hosted", level="WARNING") as logs:
                with self.assertRaises(GeminiNotConfigured):
                    Gemini(HostedSettings()).generate(MESSAGES)
            self.assertIn("reason=missing_credentials", logs.output[0])
            self.assertNotIn(MESSAGES[1]["content"], logs.output[0])
            os.environ["GOOGLE_API_KEY"] = "test-google"
            self.assertEqual(CloudSettings.from_mapping({}).provider.api_key, "test-google")
            os.environ["GEMINI_API_KEY"] = "test-gemini"
            self.assertEqual(CloudSettings.from_mapping({}).provider.api_key, "test-gemini")
            self.assertEqual(CloudSettings.from_mapping({"api_key": "test-secret"}).provider.api_key, "test-secret")

    def test_api_diagnostics_include_category_and_status_without_provider_message(self):
        for status, reason, error in (
            (400, "request_rejected", ProviderUnavailable),
            (403, "permission", GeminiAuthenticationFailed),
            (404, "model_or_endpoint_unavailable", GeminiModelUnavailable),
            (429, "quota", RateLimited),
        ):
            provider, requests = self.provider([status])
            with self.subTest(status=status), self.assertLogs("backend.hosted", level="WARNING") as logs:
                with self.assertRaises(error) as raised:
                    provider.generate(MESSAGES)
            self.assertEqual(len(requests), 1)
            self.assertIn(f"http_status={status}", logs.output[0])
            self.assertIn(f"reason={reason}", logs.output[0])
            self.assertNotIn("private provider detail", logs.output[0] + str(raised.exception))
            self.assertNotIn("fake-test-key", logs.output[0] + str(raised.exception))
        rejected, _ = self.provider([httpx.Response(400, json={"error": {
            "message": "fake-test-key", "details": [{"reason": "API_KEY_INVALID", "metadata": {"key": "fake-test-key"}}],
        }})])
        with self.assertLogs("backend.hosted", level="WARNING") as logs:
            with self.assertRaises(GeminiAuthenticationFailed):
                rejected.generate(MESSAGES)
        self.assertIn("reason=authentication", logs.output[0])
        self.assertNotIn("fake-test-key", logs.output[0])

    def provider(self, replies, *, limits=None):
        requests = []

        def transport(request):
            requests.append((request.url.path, json.loads(request.read())))
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, httpx.Response):
                return reply
            if reply != 200:
                return httpx.Response(reply, json={"error": {"code": reply, "message": "private provider detail"}})
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [
                {"text": '{"claims":[],"incomplete":true}'}], "role": "model"}, "finishReason": "STOP"}]})

        client = genai.Client(api_key="fake-test-key", vertexai=False, http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(transport)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        self.addCleanup(client.close)
        return Gemini(limits or HostedSettings(), client=client, pause=lambda seconds: None), requests

    def test_official_sdk_request_is_structured_and_has_no_tools(self):
        provider, requests = self.provider([200])
        raw, model = provider.generate(MESSAGES)
        self.assertEqual(json.loads(raw)["claims"], [])
        self.assertEqual(model, DEFAULT_MODEL)
        path, body = requests[0]
        self.assertTrue(path.endswith(f"{DEFAULT_MODEL}:generateContent"))
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertIn("responseJsonSchema", body["generationConfig"])
        self.assertNotIn("tools", body)
        self.assertNotIn("fake-test-key", repr(provider.settings))

    def test_429_is_not_retried_and_503_retries_once(self):
        limited, requests = self.provider([429])
        with self.assertRaisesRegex(RateLimited, "quota"):
            limited.generate(MESSAGES)
        self.assertEqual(len(requests), 1)
        retry, requests = self.provider([503, 200])
        retry.generate(MESSAGES)
        self.assertEqual(len(requests), 2)
        unavailable, requests = self.provider([503, 503])
        with self.assertRaisesRegex(ProviderUnavailable, "temporarily unavailable"):
            unavailable.generate(MESSAGES)
        self.assertEqual(len(requests), 2)

    def test_transport_timeout_is_not_retried(self):
        provider, requests = self.provider([httpx.ReadTimeout("private timeout")])
        with self.assertRaisesRegex(GenerationTimeout, "timed out"):
            provider.generate(MESSAGES)
        self.assertEqual(len(requests), 1)

    def test_budget_bounds_retries_and_resets_after_window(self):
        retry, requests = self.provider([503], limits=HostedSettings(requests_per_minute=1))
        with self.assertRaises(RateLimited):
            retry.generate(MESSAGES)
        self.assertEqual(len(requests), 1)
        now = [0]
        budget = RequestBudget((60, 1), clock=lambda: now[0])
        budget.acquire()
        with self.assertRaises(RateLimited):
            budget.acquire()
        now[0] = 60
        budget.acquire()

    def test_input_and_busy_guards(self):
        service = DemoService(CloudSettings())
        with self.assertRaises(ValueError):
            service.ask("x" * 1601, [], {})
        self.assertTrue(service.slot.acquire(blocking=False))
        try:
            with self.assertRaises(Busy):
                service.ask("What is the weather in Pune?", [], {})
        finally:
            service.slot.release()
        self.assertEqual(service.ask("What is the weather in Pune?", [], {})["status"], "out_of_scope")
        self.assertIsNone(service.assistant.retriever)
        with self.assertRaises(ValueError):
            HostedSettings(model="gemini-2.5-flash")


class BundleTests(unittest.TestCase):
    def test_local_default_restores_from_sidecar_and_caches_the_index(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            np.save(root / "row.npy", np.ones((1, 384), dtype="<f4"))
            path = root / LOCAL_BUNDLE
            path.parent.mkdir()
            with ZipFile(path, "w") as archive:
                archive.writestr("manifest.json", json.dumps({
                    "schema": 1, "collection": COLLECTION_NAME, "chunks": 1, "dimension": 384,
                    "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
                }))
                archive.writestr("documents.jsonl", json.dumps({
                    "id": "row", "document": "A verified passage", "metadata": {"pdf_page": 7},
                }) + "\n")
                archive.write(root / "row.npy", "vectors.npy")
                archive.writestr("minilm.onnx", "test fixture")
                archive.writestr("tokenizer.json", "{}")
                archive.writestr("MiniLM-APACHE-2.0.txt", "")
                archive.writestr("MiniLM-NOTICE.txt", "")
            digest = sha256(path)
            with self.assertRaises(IndexUnavailable):
                prepare_bundle("0" * 64, local=path, cache_dir=root)
            sidecar = path.with_suffix(".sha256")
            with patch("backend.db.PROJECT_ROOT", root), patch("backend.index_bundle.download_bundle") as download:
                with self.assertRaisesRegex(IndexUnavailable, "sidecar"):
                    prepare_bundle("", cache_dir=root)
                sidecar.write_text("f" * 64, encoding="ascii")
                with self.assertRaisesRegex(IndexUnavailable, "checksum"):
                    prepare_bundle("", cache_dir=root)
                sidecar.write_text(digest.upper() + "\n", encoding="ascii")
                snapshot, count = prepare_bundle("", cache_dir=root)
                self.assertEqual(prepare_bundle("", local=LOCAL_BUNDLE, cache_dir=root), (snapshot, 1))
                download.assert_not_called()
            self.assertEqual(count, 1)
            self.assertTrue((snapshot / "chroma.sqlite3").is_file())
            path.unlink()
            self.assertEqual(prepare_bundle(digest, local=path, cache_dir=root), (snapshot, 1))

    def test_explicit_release_url_takes_priority_over_automatic_local_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / LOCAL_BUNDLE
            path.parent.mkdir()
            path.touch()
            url = "https://github.com/example/project/releases/download/v1/legalbot-index.zip"
            with patch("backend.db.PROJECT_ROOT", root), patch("backend.index_bundle.download_bundle") as download:
                download.side_effect = IndexUnavailable("Simulated release error")
                with self.assertRaisesRegex(IndexUnavailable, "Simulated release error"):
                    prepare_bundle("0" * 64, url=url, cache_dir=root)
                self.assertEqual(download.call_args.args[0], url)

    def test_rejects_unexpected_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "unsafe.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("../unexpected", "no")
            with self.assertRaises(IndexUnavailable):
                prepare_bundle(sha256(path), local=path, cache_dir=root)
            self.assertFalse((root / "unexpected").exists())
