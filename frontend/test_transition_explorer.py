import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.hosted import RateLimited


class TransitionExplorerTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def test_search_columns_official_sources_and_no_result(self):
        app = AppTest.from_file(str(self.path), default_timeout=20).run()
        app.radio[0].set_value("Law Transition Explorer").run()
        app.selectbox[0].set_value("BSA").run()
        app.text_input[0].set_value("IEA 65B").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Previous law" == title.value for title in app.subheader))
        self.assertTrue(any("Successor law" == title.value for title in app.subheader))
        labels = [source.label for source in app.expander]
        self.assertTrue(any("IEA · Section 65B(1) · PDF p. 34" in label for label in labels))
        self.assertTrue(any("BSA · Section 63(4) · PDF p. 23" in label for label in labels))
        app.text_input[0].set_value("BSA 65B").run()
        self.assertTrue(any("No verified mapping" in message.value for message in app.info))
        self.assertFalse(app.exception)

    def test_hosted_failure_preserves_citations_and_qa_history(self):
        service = Mock()
        service.ask.return_value = {"answer": "Verified answer.", "status": "relevant"}
        service.explain_transition.side_effect = RateLimited("quota")
        with patch("frontend.app.DemoService", return_value=service):
            app = AppTest.from_file(str(self.path), default_timeout=20).run()
            app.chat_input[0].set_value("When is an RTI reply due?").run()
            app.radio[0].set_value("Law Transition Explorer").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("temporarily unavailable" in item.value for item in app.info))
            self.assertTrue(any("PDF p. 19" in source.label for source in app.expander))
            app.radio[0].set_value("Legal Q&A").run()
            self.assertEqual(app.session_state.turns[0][0], "When is an RTI reply due?")
            self.assertTrue(any("Verified answer." in item.value for item in app.markdown))
            self.assertFalse(app.exception)
