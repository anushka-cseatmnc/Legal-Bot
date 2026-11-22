import os
import tempfile
import unittest
from pathlib import Path
from time import sleep
from unittest.mock import Mock, patch

import streamlit as st
from streamlit import config as st_config
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest

from backend.cloud_service import CloudSettings, create_retriever
from backend.db import project_path
from backend.index_bundle import LOCAL_BUNDLE


class StreamlitTests(unittest.TestCase):
    def test_secrets_file_added_and_corrected_without_reusing_unconfigured_client(self):
        from frontend.app import InvalidSecrets, service

        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        get_option = st_config.get_option
        native_secrets = Secrets()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secrets.toml"
            with (
                patch("frontend.app.st.secrets", native_secrets),
                patch.object(Secrets, "_maybe_install_file_watchers"),
                patch.object(st_config, "get_option", side_effect=lambda name: (
                    [str(path)] if name == "secrets.files" else get_option(name)
                )),
                patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}),
            ):
                original = service()
                self.assertIsNone(original.assistant.provider.client)
                path.write_text("[legalbot]\napi_key = fake-local-test-key\n", encoding="utf-8")
                with self.assertLogs("frontend.app") as logs, self.assertRaises(InvalidSecrets):
                    service()
                self.assertIn("invalid_toml", logs.output[0])
                self.assertNotIn("fake-local-test-key", logs.output[0])
                path.write_text('[legalbot]\napi_key = "fake-local-test-key"\n', encoding="utf-8")
                updated = service()
                self.assertIsNot(updated, original)
                provider = updated.assistant.provider
                self.assertIsNotNone(provider.client)
                self.addCleanup(provider.client.close)
                self.assertTrue(updated.config.provider.api_key == provider.client._api_client.api_key)
                self.assertIs(service(), updated)

    def test_answer_and_source_render_with_session_scope(self):
        backend = Mock()
        backend.ask.return_value = {
            "answer": "First paragraph.\n\nLast paragraph. [S1]",
            "status": "relevant",
            "citations": [{"id": "S1", "act": "Right to Information Act",
                           "document": "rti.pdf", "section": "7", "pdf_page": 7,
                           "page_label": "7", "excerpt": "Evidence text.", "quotes": []}],
        }
        path = Path(__file__).with_name("app.py")
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with patch("backend.cloud_service.DemoService", return_value=backend):
            app = AppTest.from_file(str(path)).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("What is the RTI deadline?").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Last paragraph." in markup.value for markup in app.markdown))
            citation = next(expander for expander in app.expander if expander.label.startswith("[S1]"))
            self.assertIn("Section 7 · PDF p. 7", citation.label)
            app.chat_input[0].set_value("What about life or liberty?").run()
            self.assertEqual(backend.ask.call_args.args[1], ["What is the RTI deadline?"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("What about life or liberty?").run()
            self.assertEqual(backend.ask.call_args.args[1], [])
            second_session = AppTest.from_file(str(path)).run()
            second_session.chat_input[0].set_value("Another question").run()
            self.assertEqual(backend.ask.call_args.args[1], [])

    @unittest.skipUnless(
        (os.getenv("LEGALBOT_INDEX_SHA256") and os.getenv("LEGALBOT_INDEX_BUNDLE"))
        or (project_path(LOCAL_BUNDLE).is_file() and project_path(LOCAL_BUNDLE).with_suffix(".sha256").is_file()),
        "Provide the full release bundle and its checksum.",
    )
    def test_published_entry_point_with_real_index(self):
        path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        app = AppTest.from_file(str(path)).run()
        self.assertFalse(app.exception)
        app.toggle[0].set_value(True).run()
        app.chat_input[0].set_value("When is an RTI reply due?").run(timeout=120)
        first = app.session_state.turns[-1][1]
        self.assertIn("thirty days", first["answer"].lower())
        self.assertTrue(any("Section 7 · PDF p. 7" in source.label for source in app.expander))
        sleep(2.05)
        app.chat_input[0].set_value("What if my life is at risk?").run()
        follow_up = app.session_state.turns[-1][1]
        self.assertIn("forty-eight hours", follow_up["answer"].lower())
        self.assertEqual(follow_up["citations"][0]["pdf_page"], 7)
        sleep(2.05)
        app.chat_input[0].set_value("What is the weather in Pune tomorrow?").run()
        self.assertEqual(app.session_state.turns[-1][1]["status"], "out_of_scope")
        self.assertFalse(app.exception)

        retriever = create_retriever(CloudSettings.from_mapping({}))
        theft = retriever.retrieve("What does BNS section 303 say about theft?")
        self.assertEqual(theft.status, "relevant")
        self.assertTrue(any(
            doc.metadata.get("document") == "BNS 2023.pdf"
            and doc.metadata.get("section") == "303"
            and doc.metadata.get("pdf_page") == 78
            and "dishonestly any movable property" in " ".join(doc.page_content.lower().split())
            for doc in theft.documents
        ))
