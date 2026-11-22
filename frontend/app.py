import logging
from collections.abc import Mapping
from datetime import date

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from toml import TomlDecodeError

from backend.cloud_service import MAX_HISTORY, MAX_QUESTION, CloudSettings, DemoService, session_gate
from backend.db import IndexUnavailable
from backend.hosted import (
    Busy, GeminiAuthenticationFailed, GeminiModelUnavailable, GeminiNotConfigured, RateLimited,
)
from backend.rag_pipeline import GenerationTimeout, ProviderUnavailable
from frontend.transition_explorer import show_explorer

logger = logging.getLogger(__name__)
EXAMPLES = (
    "What is the punishment for theft under BNS Section 303?",
    "What minimum ages does the Hindu Marriage Act prescribe?",
    "When is an RTI reply due?",
)


class InvalidSecrets(ProviderUnavailable):
    """Streamlit found a secrets file that it could not parse."""


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_service(config: CloudSettings):
    return DemoService(config)


def service():
    try:
        settings = st.secrets.get("legalbot", {})
    except StreamlitSecretNotFoundError as exc:
        if isinstance(exc.__context__, (TomlDecodeError, UnicodeError)):
            logger.warning("Streamlit secrets rejected reason=invalid_toml")
            raise InvalidSecrets("Streamlit secrets contains invalid TOML.") from None
        settings = {}
    if not isinstance(settings, Mapping):
        logger.warning("Streamlit secrets rejected reason=invalid_legalbot_table")
        raise InvalidSecrets("Streamlit secrets needs a [legalbot] table.")
    return cached_service(CloudSettings.from_mapping(settings))


def ask(question, history, context, budget):
    try:
        return service().ask(question, history, context, budget)
    except RateLimited:
        message = "The demo or its hosted provider has reached a usage limit. Please try again later."
    except Busy:
        message = "LegalBot is answering another question. Please try again shortly."
    except GenerationTimeout:
        message = "The hosted model took too long to respond. Please try again."
    except GeminiNotConfigured:
        message = "Gemini is not configured for this app. Verified statutory lookups are still available."
    except GeminiAuthenticationFailed:
        message = "Gemini rejected the app's credentials or permissions. Verified statutory lookups are still available."
    except GeminiModelUnavailable:
        message = "The configured Gemini model is unavailable. Verified statutory lookups are still available."
    except InvalidSecrets:
        message = "The app's secrets settings are invalid. Check the TOML syntax in the [legalbot] section."
    except ProviderUnavailable:
        message = "Hosted explanations are temporarily unavailable. A specific statutory lookup may still work."
    except IndexUnavailable:
        message = "The legal index is temporarily unavailable. Please try again later."
    except Exception as exc:
        logger.warning("Request failed (%s)", type(exc).__name__)
        message = "LegalBot could not process this question. Please try again later."
    return {"answer": message, "status": "error"}


def show_answer(result):
    status = result.get("status")
    if status == "error":
        st.error(result["answer"])
        return
    citations = result.get("citations", [])
    general = result.get("general_information")
    if citations:
        st.caption("Partially Verified" if status == "partial" or general else "Verified")
    elif not general and status in {"insufficient", "partial", "out_of_scope"}:
        st.caption({"insufficient": "Insufficient evidence", "partial": "Insufficient primary evidence",
                    "out_of_scope": "Outside the corpus scope"}[status])
    if result.get("answer_kind") != "general":
        if result.get("answer_kind") == "excerpt":
            st.text(result["answer"])
        else:
            st.markdown(result["answer"], unsafe_allow_html=False)
    if general:
        st.caption("General AI Information — Not Verified Against LegalBot's Documents.")
        st.caption("Outside LegalBot’s legal specialization. Based on general AI knowledge."
                   if general["outside_legal"] else "Based on general AI knowledge. Verify current legal requirements with official sources.")
        st.text(general["text"])
    groups = {}
    for source in citations:
        passages = tuple(dict.fromkeys(" ".join(text.split()) for text in (
            source.get("quotes") or [source["excerpt"]]
        ) if text.strip()))
        key = (source["document"], source["pdf_page"], source.get("section"), source.get("subsection"), passages)
        groups.setdefault(key, []).append(source)
    for group, items in groups.items():
        source = items[0]
        identifiers = ", ".join(f"[{item['id']}]" for item in items)
        label = f"{identifiers} {source.get('act') or source['document']}"
        if source.get("section"):
            kind = "Article" if source.get("provision_type") == "article" else "Section"
            label += f" · {kind} {source['section']}"
            if source.get("subsection"):
                label += f"({source['subsection']})"
        label += f" · PDF p. {source['pdf_page']}"
        with st.expander(label):
            st.text(source["document"])
            if source.get("section_heading"):
                st.text(source["section_heading"])
            if source.get("page_label") and source["page_label"] != str(source["pdf_page"]):
                st.caption(f"PDF page label {source['page_label']}")
            if source.get("effective_from"):
                st.caption(f"Effective from {source['effective_from']}")
            if source.get("legal_status") == "legacy_subject_to_savings":
                st.caption(f"Historical text. Savings provision: {source.get('savings') or 'see the applicable law'}.")
            elif source.get("legal_status") == "unverified":
                st.caption("Current status of this edition is unverified.")
            for passage in group[-1]:
                st.text(passage)
            if (source.get("source_url") or "").startswith("https://"):
                st.link_button("Official source", source["source_url"])
            if (source.get("commencement_source") or "").startswith("https://"):
                st.link_button("Commencement source", source["commencement_source"])


def show_chat():
    with st.sidebar:
        verified_only = st.toggle(
            "Verified Sources Only", key="verified_only", value=st.session_state.get("verified_mode", False),
            help="Use only indexed evidence for new questions.",
        )
        st.session_state.verified_mode = verified_only
        if st.button("New conversation"):
            st.session_state.turns = []
            st.session_state.pop("event_date", None)
            st.session_state.pop("procedural_status", None)
            st.rerun()
        with st.expander("Case details"):
            event_date = st.date_input("Incident date", value=None, min_value=date(1860, 1, 1),
                                       max_value=date.today(), key="event_date")
            labels = {None: "Unspecified", "pending_at_transition": "Pending on 1 July 2024",
                      "not_pending_at_transition": "No proceeding pending on 1 July 2024", "unsure": "Unsure"}
            procedural_status = st.selectbox("Procedural status", list(labels), format_func=labels.get,
                                             key="procedural_status")
    turns = st.session_state.setdefault("turns", [])
    for question, result in turns:
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            show_answer(result)

    example = st.pills("Try a question", EXAMPLES, key="example_question") if not turns else None
    question = st.chat_input("Ask about Indian law", max_chars=MAX_QUESTION) or example
    if question and question.strip():
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking legal sources…"):
                context = [prior for prior, result in turns if result.get("status") != "error"]
                result = ask(question, context[-MAX_HISTORY:], {
                    "event_date": event_date.isoformat() if event_date else None,
                    "procedural_status": procedural_status,
                    "allow_general": not verified_only,
                }, st.session_state.request_budget)
            show_answer(result)
        turns.append((question, result))
        del turns[:-20]


def main():
    st.set_page_config(page_title="LegalBot", page_icon="⚖", layout="centered")
    st.title("LegalBot")
    st.caption("Indian legal information with citations to the underlying documents.")
    st.info("For education only. Please do not share personal or confidential details. Questions that need "
            "a generated explanation are sent to Google Gemini.")
    if "request_budget" not in st.session_state:
        st.session_state.request_budget = session_gate()
    section = st.radio("LegalBot section", ("Legal Q&A", "Law Transition Explorer"), horizontal=True,
                       label_visibility="collapsed")
    if section == "Legal Q&A":
        show_chat()
    else:
        show_explorer(service, st.session_state.request_budget)


if __name__ == "__main__":
    main()
