"""Verified statutory comparisons and preliminary commencement guidance."""

import json
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import urlsplit

from langchain_core.documents import Document

from backend.db import PROJECT_ROOT, IndexUnavailable
from backend.rag_pipeline import SYSTEM, validate_claims
from backend.retrieval import ACTS, TRANSITION, act_ids, tokens

PAIRS = {"BNS": ("IPC", "BNS"), "BNSS": ("CrPC", "BNSS"), "BSA": ("IEA", "BSA")}
KINDS = {
    "renumbered": "Renumbered",
    "modified": "Modified",
    "introduced": "Newly introduced",
    "omitted": "Omitted as a standalone provision",
    "non_equivalent": "No straightforward one-to-one equivalent",
}
SECTION = re.compile(r"([1-9][0-9]{0,2}[a-z]{0,2})(?:\s*\(\s*([0-9]+[a-z]?)\s*\))?", re.I)
COMMENCEMENT = date.fromisoformat(TRANSITION)


@dataclass(frozen=True)
class Passage:
    id: str
    act_id: str
    ref: str
    page: int
    text: str

    @property
    def section(self):
        return self.ref.split("(")[0]


@dataclass(frozen=True)
class Mapping:
    id: str
    pair: str
    title: str
    keywords: tuple[str, ...]
    kind: str
    old: tuple[str, ...]
    new: tuple[str, ...]
    notes: tuple[dict, ...]
    verification: str
    exception: str | None = None


@dataclass(frozen=True)
class Catalog:
    acts: dict
    pairs: dict
    passages: dict[str, Passage]
    mappings: dict[str, Mapping]


@lru_cache(maxsize=1)
def load_catalog():
    raw = json.loads((PROJECT_ROOT / "data" / "law_transitions.json").read_text(encoding="utf-8"))
    passages = {key: Passage(key, **entry) for key, entry in raw["passages"].items()}
    mappings = {}
    for row in raw["mappings"]:
        mapping = Mapping(**{**row, **{key: tuple(row[key]) for key in ("old", "new", "notes", "keywords")}})
        old_act, new_act = PAIRS[mapping.pair]
        evidence = set(mapping.old + mapping.new)
        if (mapping.id in mappings or mapping.verification != "verified" or mapping.kind not in KINDS
                or bool(mapping.old) != (mapping.kind != "introduced")
                or bool(mapping.new) != (mapping.kind != "omitted")
                or any(passages[key].act_id != old_act for key in mapping.old)
                or any(passages[key].act_id != new_act for key in mapping.new)
                or any(not note["sources"] or set(note["sources"]) - evidence for note in mapping.notes)):
            raise ValueError(f"Invalid transition mapping {mapping.id}")
        mappings[mapping.id] = mapping
    for passage in passages.values():
        if not SECTION.fullmatch(passage.ref) or passage.page < 1 or not passage.text.strip():
            raise ValueError(f"Invalid source {passage.id}")
    for details in raw["acts"].values():
        if urlsplit(details["url"]).scheme != "https":
            raise ValueError("An official statute link must use HTTPS.")
    for act, pair in raw["pairs"].items():
        if (pair["effective_from"] != TRANSITION
                or any(passages[key].act_id != act for key in (pair["savings"], *pair.get("savings_continuation", ())))):
            raise ValueError("Invalid transitional authority.")
    return Catalog(raw["acts"], raw["pairs"], passages, mappings)


def search(catalog, pair, query):
    """Return only catalog entries. An exact section never falls back to a fuzzy topic."""
    query = query.strip()
    choices = [mapping for mapping in catalog.mappings.values() if mapping.pair == pair]
    if not query:
        return choices
    acts = act_ids(query)
    if acts and not acts.intersection(PAIRS[pair]):
        return []
    remainder = query
    for act in acts:
        remainder = re.sub(ACTS[act][0], "", remainder, flags=re.I)
    remainder = re.sub(r"\b(?:1860|1872|1973|2023|the|sections?|of)\b|\bs\.|[§,]", "", remainder, flags=re.I).strip()
    if not remainder:
        return choices if acts else []
    if match := SECTION.fullmatch(remainder):
        section, sub = match.groups()
        ref = f"{section.upper()}({sub.lower()})" if sub else section.upper()
        return [
            mapping for mapping in choices
            if any((not acts or source.act_id in acts)
                   and (source.ref == ref or not sub and source.section == ref)
                   for source in (catalog.passages[key] for key in mapping.old + mapping.new))
        ]
    if any(char.isdecimal() for char in remainder):
        return []
    terms = set(tokens(remainder))
    return [
        mapping for mapping in choices if terms and any(
            (not acts or any(catalog.passages[key].act_id in acts for key in mapping.old + mapping.new))
            and terms <= set(tokens(label))
            for label in (mapping.title, *mapping.keywords)
        )
    ]


@dataclass(frozen=True)
class Applicability:
    status: str
    text: str
    sources: tuple[str, ...]


def applicability(catalog, mapping, event, pending, jurisdiction):
    pair = catalog.pairs[mapping.pair]
    sources = (pair["savings"], *pair.get("savings_continuation", ()))
    if jurisdiction == "outside":
        return Applicability("outside", "This comparison assumes Indian law. Cross-border facts or a foreign forum require a separate jurisdiction assessment.", ())
    if jurisdiction == "unsure":
        return Applicability("clarify", "Identify the relevant Indian jurisdiction or forum before applying the transition rules.", sources)
    if mapping.pair == "BNSS" and jurisdiction == "special":
        return Applicability(
            "exception", "BNSS section 1(2) limits application in Nagaland and the defined tribal areas. Chapters IX, XI and XII are treated separately. Check the location, relevant chapter and any State notification.",
            ("BNSS-1", *sources),
        )
    if mapping.pair == "BNS":
        if event and event < COMMENCEMENT:
            return Applicability("legacy", "For alleged conduct before 1 July 2024, start with the law in force at the time. BNS section 358 preserves prior IPC liabilities and proceedings. Article 20(1) also limits retrospective conviction and heavier penalties.", (*sources, "CONS-20"))
        if mapping.exception == "bns_106_2":
            return Applicability("exception", "S.O. 850(E) excluded BNS section 106(2) from the 1 July 2024 commencement. S.O. 848(E) also excluded its BNSS First Schedule entry. The sources here do not establish a later commencement. Verify the notification history and the date of the alleged conduct.", sources)
        if event is None:
            return Applicability("clarify", "Enter the date of the alleged offence. The date a proceeding began does not by itself choose the substantive offence law.", sources)
        return Applicability("successor", "For alleged conduct on or after 1 July 2024, the commenced BNS text is the starting point. Check the actual conduct, any continuing offence, and applicable special law.", sources)
    if pending == "pending":
        return Applicability("legacy", f"The listed matter was pending immediately before commencement. The savings provision points to the {PAIRS[mapping.pair][0]} for that saved matter. A later, distinct step may require its own analysis.", sources)
    if pending != "not_pending":
        return Applicability("clarify", "Confirm whether the specific matter was pending immediately before 1 July 2024. The underlying event date alone does not resolve this savings question.", sources)
    if event is None or event < COMMENCEMENT:
        return Applicability("conditional", "If the relevant proceeding began on or after 1 July 2024 and no listed matter was saved, start with the successor law. An earlier underlying event does not by itself decide the procedural or evidence regime.", sources)
    return Applicability("successor", "For a new matter on or after 1 July 2024 with no applicable saved proceeding, start with the successor law. Check special or local law and the specific procedural step.", sources)


def textual_diff(old, new):
    """Word differences for short, one-to-one excerpts."""
    before, after = old.split(), new.split()
    return [(tag, " ".join(before[i:j]), " ".join(after[k:l]))
            for tag, i, j, k, l in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()]


def _source(catalog, key):
    passage = catalog.passages[key]
    act = catalog.acts[passage.act_id]
    return {
        "id": key, "act": act["name"], "section": passage.section,
        "subsection": passage.ref.split("(")[1][:-1] if "(" in passage.ref else "",
        "document": act["document"], "pdf_page": passage.page, "source_url": act["url"],
        "excerpt": passage.text,
    }


def hybrid_evidence(catalog, mapping, retriever):
    """Choose indexed supporting excerpts only when their exact text matches the verified PDF slice."""
    sources = {key: _source(catalog, key) for key in mapping.old + mapping.new}
    indexed = set()
    groups = sorted({(catalog.passages[key].act_id, source["section"]) for key, source in sources.items()
                     if catalog.acts[catalog.passages[key].act_id].get("indexed")})
    available = {(doc.metadata.get("act_id"), doc.metadata.get("section")) for doc in retriever.documents.values()}
    for act, section in groups:
        if (act, section) not in available:
            continue
        result = retriever.retrieve(f"{act} section {section} {mapping.title}")
        for doc in result.documents:
            for key, source in sources.items():
                if (catalog.passages[key].act_id != act or source["section"] != section
                        or doc.metadata.get("act_id") != act or doc.metadata.get("section") != section
                        or doc.metadata.get("document") != source["document"]
                        or doc.metadata.get("pdf_page") != source["pdf_page"]):
                    continue
                excerpt = " ".join(doc.page_content.split())
                if len(excerpt) >= 60 and (excerpt in source["excerpt"] or source["excerpt"] in excerpt):
                    indexed.add(key)
    ordered = {}
    for side in (mapping.old, mapping.new):
        for key in side:
            if key in indexed:
                ordered[key] = sources[key]
        supplemental = [Document(id=key, page_content=sources[key]["excerpt"]) for key in side if key not in indexed]
        for doc in retriever.rank_supplemental(mapping.title, supplemental):
            ordered[doc.id] = sources[doc.id]
    return ordered


def generate_explanation(catalog, mapping, retriever_factory, provider):
    """Add cited plain-English notes to the authoritative editorial comparison."""
    if not mapping.old or not mapping.new:
        return (), "Hosted comparison needs verified text on both sides."
    try:
        sources = hybrid_evidence(catalog, mapping, retriever_factory())
    except IndexUnavailable:
        sources = {key: _source(catalog, key) for key in mapping.old + mapping.new}
    # The catalog bounds this to a few exact excerpts. Visitor facts stay local.
    evidence = {key: {**source, "excerpt": source["excerpt"][:900]} for key, source in sources.items()}
    if any(not source["excerpt"].strip() for source in evidence.values()):
        return (), "Supporting text is missing. The hosted comparison was skipped."
    instruction = (
        "\nProvide separate plain-English explanations of at least one passage from each side. "
        "Each claim must describe only what its single cited passage says. "
        "Do not assert equivalence, evaluate applicability, or infer differences beyond the curated comparison."
    )
    content = json.dumps({"topic": mapping.title, "old": mapping.old, "new": mapping.new, "evidence": evidence})
    raw, _ = provider.generate([{"role": "system", "content": SYSTEM + instruction}, {"role": "user", "content": content}])
    claims, incomplete = validate_claims(raw, evidence)
    cited = {key for _, key, _ in claims}
    if not cited.intersection(mapping.old) or not cited.intersection(mapping.new):
        return (), "The hosted model did not produce a verifiable explanation for both sides."
    return tuple(claims), "The hosted explanation is partial." if incomplete else None
