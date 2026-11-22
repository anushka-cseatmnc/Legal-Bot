"""Streamlit UI for the verified Law Transition Explorer."""

import logging
from datetime import date
from html import escape
from pathlib import PurePath

import streamlit as st

from backend.hosted import Busy, RateLimited
from backend.rag_pipeline import GenerationTimeout, ProviderUnavailable
from backend.transition import KINDS, PAIRS, applicability, load_catalog, search, textual_diff

logger = logging.getLogger(__name__)
PAIR_LABELS = {"BNS": "IPC → BNS", "BNSS": "CrPC → BNSS", "BSA": "Evidence Act → BSA"}
JURISDICTIONS = {
    "unsure": "Select a jurisdiction",
    "india": "India · general central-law rules",
    "special": "Nagaland or BNSS-defined tribal area",
    "outside": "Foreign forum or cross-border jurisdiction",
}
PENDING = {
    "unsure": "Unsure",
    "pending": "The relevant matter was pending",
    "not_pending": "The relevant matter was not pending",
}


def citation(catalog, key):
    passage = catalog.passages[key]
    act = catalog.acts[passage.act_id]
    unit = "Article" if passage.act_id == "CONS" else "Section"
    with st.expander(f"[{key}] {passage.act_id} · {unit} {passage.ref} · PDF p. {passage.page}"):
        st.caption(act["name"])
        st.text(PurePath(act["document"]).name)
        st.text(passage.text)
        url = act["url"]
        if act["page_links"]:
            url += f"#page={passage.page}"
        st.link_button("Open official source", url)


def compare_columns(catalog, mapping):
    old, new = st.columns(2, gap="medium")
    for column, (side, act_id, keys) in zip(
        (old, new), (("Previous law", PAIRS[mapping.pair][0], mapping.old),
                     ("Successor law", PAIRS[mapping.pair][1], mapping.new))
    ):
        with column:
            st.subheader(side)
            st.caption(catalog.acts[act_id]["name"])
            if not keys:
                st.info("No standalone counterpart in the verified correspondence.")
            for key in keys:
                passage = catalog.passages[key]
                st.markdown(f"**Section {passage.ref}** · PDF p. {passage.page}")
                st.text(passage.text)


def show_diff(catalog, mapping):
    if len(mapping.old) != 1 or len(mapping.new) != 1:
        return
    old, new = (catalog.passages[key].text for key in (*mapping.old, *mapping.new))
    changes = textual_diff(old, new)
    if all(tag == "equal" for tag, _, _ in changes):
        st.caption("The selected excerpts have identical wording.")
        return
    with st.expander("Word differences in the displayed excerpts"):
        st.caption("Strikethrough marks removed wording. Underlining marks added wording. This comparison covers the excerpts.")
        before = " ".join(escape(word) if tag == "equal" else f"<del>{escape(word)}</del>"
                          for tag, word, _ in changes if word)
        after = " ".join(escape(word) if tag == "equal" else f"<ins>{escape(word)}</ins>"
                         for tag, _, word in changes if word)
        left, right = st.columns(2)
        left.html(f"<p>{before}</p>")
        right.html(f"<p>{after}</p>")


def show_applicability(catalog, mapping):
    with st.expander("Check preliminary applicability"):
        st.caption("Use general dates and status only. Consult a qualified lawyer for advice about a specific case.")
        jurisdiction = st.selectbox("Jurisdiction", JURISDICTIONS, format_func=JURISDICTIONS.get)
        label = "Date of alleged offence" if mapping.pair == "BNS" else "Date of relevant event or procedural step"
        event = st.date_input(label, value=None, max_value=date.today(), min_value=date(1860, 1, 1),
                              key=f"transition_date_{mapping.pair}")
        pending = None
        if mapping.pair != "BNS":
            label = "Status immediately before 1 July 2024"
            if mapping.pair == "BNSS":
                st.caption("Check the relevant appeal, application, trial, inquiry or investigation.")
            else:
                st.caption("Check the relevant application, trial, inquiry, investigation, proceeding or appeal.")
            pending = st.selectbox(label, PENDING, format_func=PENDING.get, key=f"transition_pending_{mapping.pair}")
        result = applicability(catalog, mapping, event, pending, jurisdiction)
        st.info(result.text)
    return result.sources


def hosted_explanation(service, mapping, budget):
    if not mapping.old or not mapping.new:
        return
    if st.button("Explain the provisions with Gemini"):
        try:
            with st.spinner("Checking the indexed evidence and requesting an explanation…"):
                result = service().explain_transition(mapping.id, budget)
        except (Busy, RateLimited, GenerationTimeout, ProviderUnavailable):
            result = (), "Hosted explanation is temporarily unavailable. The verified comparison and sources remain available."
        except Exception as exc:
            logger.warning("Transition explanation failed (%s)", type(exc).__name__)
            result = (), "Hosted explanation is temporarily unavailable."
        st.session_state.transition_explanation = mapping.id, result
    previous = st.session_state.get("transition_explanation")
    if previous and previous[0] == mapping.id:
        claims, message = previous[1]
        if claims:
            st.subheader("Gemini explanation of the cited excerpts")
        for claim, source, quote in claims:
            st.text(f"{claim} [{source}]")
            with st.expander(f"Supporting quote [{source}]"):
                st.text(quote)
        if message:
            st.info(message)


def show_explorer(service, budget):
    catalog = load_catalog()
    st.header("Law Transition Explorer")
    st.caption("Compare selected provisions from India’s criminal-law transition on 1 July 2024.")
    pair = st.selectbox("Act pair", PAIR_LABELS, format_func=PAIR_LABELS.get)
    query = st.text_input("Search by Act, section or topic", placeholder="For example, theft or IPC 378", max_chars=120)
    matches = search(catalog, pair, query)
    if not matches:
        st.info("No verified mapping is available for this search. Try an Act name, section number or a different topic.")
        return
    mapping_id = st.selectbox("Verified comparison", [mapping.id for mapping in matches],
                              format_func=lambda key: catalog.mappings[key].title)
    mapping = catalog.mappings[mapping_id]
    st.subheader(mapping.title)
    st.caption(f"Verified correspondence · {KINDS[mapping.kind]}")
    st.caption("Selected statutory excerpts")
    compare_columns(catalog, mapping)
    show_diff(catalog, mapping)
    st.subheader("What changed")
    for note in mapping.notes:
        st.markdown(f"**{note['category']}**")
        st.write(note["text"] + " " + " ".join(f"[{key}]" for key in note["sources"]))
    applicability_sources = show_applicability(catalog, mapping)
    hosted_explanation(service, mapping, budget)
    st.subheader("Official sources")
    metadata = catalog.pairs[mapping.pair]
    st.link_button("NCRB correspondence table", metadata["correspondence"])
    st.link_button(f"Commencement notice · {metadata['notification']}", metadata["gazette"])
    if mapping.exception == "bns_106_2":
        st.link_button("BNSS First Schedule commencement notice", catalog.pairs["BNSS"]["gazette"])
    keys = dict.fromkeys((*mapping.old, *mapping.new, metadata["savings"], *applicability_sources))
    for key in keys:
        citation(catalog, key)
