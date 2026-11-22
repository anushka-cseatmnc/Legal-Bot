"""CPU hybrid retrieval and conservative legal applicability."""

import math
import re
from calendar import month_abbr, month_name
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from langchain_core.documents import Document

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
STOP = set(("a an and are as at be by can did do does for from had has have how i if in is it "
            "may my of on or our shall should so the their this to under was were what when "
            "where which who will with would you your usually usual generally india indian").split())
ACTS = {
    "IPC": (r"\bipc\b|indian penal code", "indian penal code 1860"),
    "CrPC": (r"\bcr\s*pc\b|code of criminal procedure", "code of criminal procedure 1973"),
    "IEA": (r"\biea\b|(?:indian )?evidence act", "indian evidence act 1872"),
    "BNS": (r"\bbns\b|bharatiya nyaya sanhita", "bharatiya nyaya sanhita 2023"),
    "BNSS": (r"\bbnss\b|bharatiya nagari(?:k|ka) suraksha sanhita", "bharatiya nagarik suraksha sanhita 2023"),
    "BSA": (r"\bbsa\b|bharatiya sakshya adhiniyam", "bharatiya sakshya adhiniyam 2023"),
    "RTI": (r"\brti\b|right to information", "right to information act"),
    "HMA": (r"hindu marriage", "hindu marriage act"),
    "SMA": (r"special marriage act", "special marriage act"),
    "CPA": (r"consumer (?:protection|complaints?|commission|dispute)", "consumer protection act"),
    "IT": (r"\bit act\b|information technology act|identity theft", "information technology act"),
    "CONS": (r"\bconstitution\b|\barticle\s+\d", "constitution of india"),
}
LEGACY = {"IPC", "CrPC", "IEA"}
SUCCESSORS = {"IPC": "BNS", "CrPC": "BNSS", "IEA": "BSA"}
CURRENT_CRIMINAL_CODES = frozenset(SUCCESSORS.values())
REFERENCE = re.compile(r"(?:\b(?:sections?|articles?|s\.)|§)\s*(\d{1,3}[a-z]{0,3})\b(?:\s*\((\d+)\))?", re.I)
EXPLICIT_SCOPE = re.compile(
    r"\b(?:california|united states|united kingdom|england|canada|australia|eu law|texas|new york)\b", re.I
)
NON_LEGAL = re.compile(r"\b(?:weather|temperature forecast|cricket score|recipe|capital of|translate this)\b", re.I)
TRANSITION = "2024-07-01"
FIR_REFUSAL = r"(?=.*\brefus\w*\b)(?=.*\b(?:police|firs?|cognizable)\b)"
FIR_ESCALATION = r"\b(?:aggrieved by a refusal|superintendent of police concerned)\b"
THEFT_PENALTY = r"\b(?:punishment|penalt(?:y|ies)|sentenc\w*|conviction|community service)\b"
RTI_URGENT = r"\blife\b.{0,24}\b(?:liberty|risk)\b"
RTI_URGENT_PASSAGE = (
    r"\bwhere the information sought for concerns the life or liberty of a person, "
    r"the same shall be provided within (?P<limit>forty[- ]eight hours) of the receipt of the request"
)
RTI_ORDINARY_PASSAGE = r"\bin any case within (?P<limit>thirty days) of the receipt of the request"
MONTHS = {name.casefold(): index for names in (month_name, month_abbr)
          for index, name in enumerate(names) if index}
EXPANSIONS = (
    (r"\bfirs?\b|(?:police.*record|record.*offence)", "information police cognizable offence cases first information report"),
    (r"\brefus\w*", "refusal superintendent police magistrate"),
    (r"\bdeadline\b|\btime limit\b", "period days limitation"),
    (r"life.*(?:risk|liberty)|(?:liberty|risk).*life", "life liberty forty eight hours"),
    (r"consumer.*(?:limit|delay|late)", "limitation period admission filed cause action sufficient cause"),
    (r"\b(?:ages?|old enough)\b.*marriage|marriage.*\bages?\b", "bride bridegroom completed age years conditions"),
    (r"identity theft", "electronic signature password unique identification"),
    (r"electronic.*(?:evidence|record)|digital evidence", "admissibility electronic record computer certificate"),
    (r"cannot.*afford.*lawyer|free legal (?:aid|help)", "free legal aid legal services"),
    (r"change.*(?:rti|information).*fees|(?:rti|information).*fees", "fee prescribed public information officer"),
)
ISSUE_FOCUS = (
    ("CPA", r"\b(?:time limit|deadline|limitation|filing delay|delay be excused)\b",
     r"\b(?:complaint unless it is filed within|not filing the complaint within such period)\b"),
    ("RTI", r"\b(?:deadline|reply|answer|response|life|liberty)\b",
     r"\b(?:disposal of request|information sought for concerns the life or liberty)\b"),
    ("BNSS", FIR_REFUSAL, FIR_ESCALATION),
    ("CrPC", r"\bfirs?\b|\bfirst information report\b", r"\binformation in cognizable cases\b"),
)


def tokens(value: str) -> list[str]:
    terms = re.findall(r"[a-z]+\d+[a-z]*|\d+[a-z]+|[a-z]+|\d+", value.casefold())
    return [term[:-1] if len(term) > 4 and term.endswith("s") and not term.endswith(("ss", "us"))
            else term for term in terms if term not in STOP]


def act_ids(text: str) -> set[str]:
    return {act for act, (pattern, _) in ACTS.items() if re.search(pattern, text, re.I)}


def source_act(doc: Document) -> str | None:
    meta = doc.metadata
    if "act_id" in meta:
        return meta["act_id"]
    return next(iter(act_ids(meta.get("act", ""))), None)


def resolve_question(question: str, history: list[str]) -> str:
    if not history or act_ids(question):
        return question
    continuation = re.search(
        r"^\s*(?:and|what if|how about|what about|then|does (?:that|it)|can (?:i|they) (?:do|appeal)|"
        r"is (?:that|it)|are there)\b|\b(?:that deadline|this case|the same|that section)\b", question, re.I
    )
    if not continuation:
        return question
    for previous in reversed(history[-4:]):
        acts = act_ids(previous)
        if acts:
            topic = ", ".join(ACTS[act][1] for act in sorted(acts))
            return f"{question} Follow-up about {topic}. Earlier question: {previous[:240]}"
        if re.search(r"\bfir\b|police refusal|cognizable offence", previous, re.I):
            return question + " In the context of recording a first information report for a cognizable offence"
    return question


def rti_deadline_pattern(question):
    if re.search(r"\b(?:appeals?|transfer|third.party|human rights|fees?)\b", question, re.I):
        return None
    if re.search(RTI_URGENT, question, re.I):
        return RTI_URGENT_PASSAGE
    if re.search(r"\b(?:deadline|time limit|response time|reply due|how long|when must)\b", question, re.I):
        return RTI_ORDINARY_PASSAGE
    return None


def date_lower_bound(value):
    if value and len(value) == 7:
        return f"{value}-01"
    if value and len(value) == 4:
        return f"{value}-01-01"
    return value


def mentioned_date(question, criminal):
    dates = re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", question)
    if len(dates) == 1:
        try:
            return date.fromisoformat(dates[0]).isoformat()
        except ValueError:
            return None
    if not criminal or dates:
        return None
    written = re.findall(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})\b", question)
    written += [(day, month, year) for month, day, year in re.findall(
        r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(20\d{2})\b", question
    )]
    if len(written) == 1 and written[0][1].casefold() in MONTHS:
        day, month, year = written[0]
        try:
            return date(int(year), MONTHS[month.casefold()], int(day)).isoformat()
        except ValueError:
            return None
    mentions = re.findall(r"\b(?:in|during|since)\s+(?:([A-Za-z]+)\s+)?(20\d{2})\b", question, re.I)
    if len(mentions) == 1:
        month, year = mentions[0]
        month_number = MONTHS.get(month.casefold()) if month else None
        if month_number:
            return f"{year}-{month_number:02d}"
        if year != "2024" and not month:
            return year
    return None


@dataclass
class LegalContext:
    question: str
    scenario: str
    acts: set[str]
    reference: str | None
    subsection: str | None
    mode: str
    event_date: str | None
    status: str | None = None
    message: str | None = None
    reference_act: str | None = None
    pending_status: str | None = None


def legal_context(question: str, history: list[str], event_date=None, jurisdiction=None, procedural_status=None):
    scenario = resolve_question(question, history)
    acts = act_ids(scenario)
    reference = REFERENCE.search(scenario)
    context = LegalContext(question, scenario, acts, reference[1].upper() if reference else None,
                           reference[2] if reference else None, "current", event_date)
    context.pending_status = procedural_status
    if reference and len(acts) > 1:
        nearby = []
        for act in sorted(acts):
            for mention in re.finditer(ACTS[act][0], scenario, re.I):
                gap = max(reference.start() - mention.end(), mention.start() - reference.end(), 0)
                if gap <= 24:
                    nearby.append((gap, act))
        if nearby:
            context.reference_act = min(nearby)[1]
    if jurisdiction and jurisdiction.upper() not in {"IN", "INDIA"} and not re.fullmatch(r"IN-[A-Z]{2}", jurisdiction.upper()):
        context.status, context.message = "out_of_scope", "This corpus covers Indian law. Please specify an Indian jurisdiction."
    elif EXPLICIT_SCOPE.search(scenario) or (NON_LEGAL.search(scenario) and not acts and not reference):
        context.status, context.message = "out_of_scope", "This question is outside the scope of the Indian legal corpus."
    if context.status:
        return context
    if reference and not acts:
        context.status, context.message = "insufficient", "Please identify the Act or code for the provision you cited."
        return context
    explicit_historical = re.search(r"\b(?:old|former|historical|repealed|under the ipc|under the crpc)\b", question, re.I)
    comparison = bool(acts & LEGACY and acts & CURRENT_CRIMINAL_CODES) or bool(
        re.search(r"\b(?:compare|comparison|transition|savings|replaced)\b", question, re.I) and acts & LEGACY
    )
    criminal = bool(acts & (LEGACY | CURRENT_CRIMINAL_CODES)) or re.search(
        r"\b(?:firs?|criminal|cognizable|offence|theft|arrest|investigation|police)\b", scenario, re.I
    )
    pending = procedural_status == "pending_at_transition"
    has_procedure = bool(acts & {"CrPC", "BNSS", "BSA", "IEA"}) or bool(
        re.search(r"\b(?:fir|police|trial|evidence|investigation|pending|procedure)\b", scenario, re.I)
    )
    if not event_date:
        context.event_date = mentioned_date(question, criminal)
    earlier = context.event_date and date_lower_bound(context.event_date) < TRANSITION
    asks_current = re.search(r"\b(?:current|currently|today|now|still in force|as of)\b", question, re.I)
    asks_own_case = re.search(r"\b(?:my (?:case|trial|fir|arrest|(?:criminal )?investigation)|i was (?:arrested|charged)|happened to me)\b", question, re.I)
    if comparison:
        context.mode = "comparison"
    elif pending and has_procedure and (reference and reference[1] in {"170", "531"}):
        context.mode = "comparison"
    elif pending and has_procedure:
        context.mode = "historical"
        context.acts = {next((old for old, new in SUCCESSORS.items() if new == act), act) for act in acts}
    elif criminal and (asks_own_case and not context.event_date and not explicit_historical or
                       earlier and has_procedure and procedural_status in {None, "unsure"} and not explicit_historical):
        context.status = "insufficient"
        context.message = ("The applicable law may depend on the incident date and whether an investigation, "
                           "trial, appeal, or other proceeding was already pending on 1 July 2024. "
                           "Please provide the date, procedural status, and relevant State or Union Territory.")
    elif criminal and (explicit_historical or earlier and not has_procedure
                       or acts & LEGACY and not asks_current and not context.event_date):
        context.mode = "historical"
    elif asks_current and acts & LEGACY:
        context.acts = {SUCCESSORS.get(act, act) for act in acts}
        if reference:
            context.status = "insufficient"
            context.message = ("The question cites a legacy section. Please give the subject of that provision "
                               "so I can locate the applicable current provision without guessing the cross-reference.")
    if context.mode == "historical" and "IEA" in context.acts and not context.status:
        context.status = "insufficient"
        context.message = "The corpus lacks an authoritative Indian Evidence Act text for this historical question."
    return context


def admissible(doc: Document, context: LegalContext) -> bool:
    meta = doc.metadata
    status = meta.get("legal_status")
    if meta.get("authority_type") == "front_matter" or meta.get("jurisdiction", "IN") != "IN":
        return False
    exception = status in {"excluded_at_commencement", "mixed_commencement"}
    direct_exception = ("BNS" in context.acts and context.reference == "106" and context.subsection == "2")
    if exception and not (direct_exception and status == "excluded_at_commencement"):
        return False
    act = source_act(doc)
    if context.mode == "current" and (status in {"legacy_subject_to_savings", "historical_reference"} or act in LEGACY):
        return False
    if context.mode == "historical" and act in SUCCESSORS.values():
        return False
    if context.pending_status == "not_pending_at_transition" and act in {"BNSS", "BSA"}:
        return True
    return not (context.mode != "comparison" and context.event_date and meta.get("effective_from")
                and date_lower_bound(context.event_date) < meta["effective_from"])


def join_overlapping_passages(documents):
    merged = []
    for doc in documents:
        following = doc.page_content.split()
        if not merged:
            merged = following
            continue
        overlap = next(
            (size for size in range(min(40, len(merged), len(following)), 3, -1)
             if merged[-size:] == following[:size]), 0,
        )
        if not overlap:
            return None
        merged.extend(following[overlap:])
    return " ".join(merged)


def complete_provision(documents, start_pattern, end_pattern, subsection):
    ordered = sorted((doc for doc in documents if doc.metadata.get("document_order") is not None),
                     key=lambda doc: doc.metadata["document_order"])
    for index, start in enumerate(ordered):
        if not re.search(start_pattern, start.page_content, re.I):
            continue
        window = ordered[index:index + 3]
        order = start.metadata["document_order"]
        if any(doc.metadata["document_order"] != order + offset
               or doc.metadata["pdf_page"] != start.metadata["pdf_page"] for offset, doc in enumerate(window)):
            continue
        complete = re.search(f"{start_pattern}.*?{end_pattern}", join_overlapping_passages(window) or "", re.I)
        if complete:
            return Document(id=start.id, page_content=complete[0],
                            metadata={**start.metadata, "subsection": subsection})
    return None


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER, revision=RERANKER_REVISION, device="cpu",
                        local_files_only=True, trust_remote_code=False, max_length=384)


class LexicalIndex:
    def __init__(self, docs):
        self.postings = defaultdict(list)
        self.words = {}
        self.length = {}
        for doc in docs:
            meta = doc.metadata
            body = tokens(doc.page_content)
            title = tokens(" ".join(str(meta.get(key, "")) for key in (
                "act", "act_id", "section_heading", "section",
            )))
            self.words[doc.id] = set(body)
            terms = body + title * 2
            self.length[doc.id] = len(terms)
            for token, count in Counter(terms).items():
                self.postings[token].append((doc.id, count))
        self.average = sum(self.length.values()) / max(len(self.length), 1) or 1

    def search(self, weights):
        scores = defaultdict(float)
        total = len(self.length)
        for token, weight in weights.items():
            postings = self.postings.get(token, ())
            idf = math.log1p((total - len(postings) + .5) / (len(postings) + .5))
            for doc_id, count in postings:
                norm = 1.2 * (.25 + .75 * self.length[doc_id] / self.average)
                scores[doc_id] += weight * idf * count * 2.2 / (count + norm)
        return scores


@dataclass
class Evidence:
    context: LegalContext
    status: str
    documents: list[Document]
    message: str | None = None


def reciprocal_ranks(*orderings):
    fused = defaultdict(float)
    for ordering in orderings:
        for rank, doc_id in enumerate(ordering, 1):
            fused[doc_id] += 1 / (40 + rank)
    return fused


class LegalRetriever:
    def __init__(self, store, *, candidates=18, top_k=5, reranker=None):
        self.store = store
        self.candidates = candidates
        self.top_k = top_k
        self.reranker = reranker
        self.documents = {}
        for offset in range(0, 1_000_000, 1000):
            rows = store.get(limit=1000, offset=offset, include=["documents", "metadatas"])
            for doc_id, text, meta in zip(rows["ids"], rows["documents"], rows["metadatas"]):
                self.documents[doc_id] = Document(id=doc_id, page_content=text, metadata=meta)
            if len(rows["ids"]) < 1000:
                break
        self.lexical = LexicalIndex(self.documents.values())
        self.by_act = {doc.id: source_act(doc) for doc in self.documents.values()}
        self.by_provision = defaultdict(list)
        for doc in self.documents.values():
            if doc.metadata.get("section"):
                self.by_provision[(doc.metadata["document"], doc.metadata["section"])].append(doc)

    def rank_supplemental(self, question, documents):
        """Rank a bounded set of verified excerpts absent from the published index."""
        if len(documents) <= 1:
            return documents
        import numpy as np

        lexical = LexicalIndex(documents)
        lexical_scores = lexical.search(dict.fromkeys(tokens(question), 1.0))
        encoder = self.store.embeddings
        query = np.asarray(encoder.embed_query(question))
        vectors = np.asarray(encoder.embed_documents([doc.page_content for doc in documents]))
        similarities = (vectors @ query) / np.maximum(np.linalg.norm(vectors, axis=1) * np.linalg.norm(query), 1e-12)
        dense = [documents[index].id for index in np.argsort(-similarities, kind="stable")]
        fused = reciprocal_ranks(sorted(lexical_scores, key=lexical_scores.get, reverse=True), dense)
        return sorted(documents, key=lambda doc: fused[doc.id], reverse=True)

    def retrieve(self, question, history=None, event_date=None, jurisdiction=None, procedural_status=None):
        context = legal_context(question, history or [], event_date, jurisdiction, procedural_status)
        if context.status:
            return Evidence(context, context.status, [], context.message)
        rti_pattern = rti_deadline_pattern(context.scenario) if context.acts == {"RTI"} else None
        if rti_pattern and context.reference in {None, "7"}:
            for (_, section), documents in self.by_provision.items():
                if section != "7":
                    continue
                for doc in documents:
                    if (self.by_act[doc.id] == "RTI" and doc.metadata.get("authority_type") == "primary_legislation"
                            and admissible(doc, context)
                            and re.search(rti_pattern, " ".join(doc.page_content.split()), re.I)):
                        return Evidence(context, "relevant", [doc])
        anchor_words = set(tokens(context.scenario))
        weights = dict.fromkeys(sorted(anchor_words), 1.0)
        for act in sorted(context.acts):
            for word in tokens(ACTS[act][1]):
                weights.setdefault(word, .6)
        expansion_words = set()
        for pattern, expansion in EXPANSIONS:
            if re.search(pattern, context.scenario, re.I):
                expansion_words.update(tokens(expansion))
        for word in sorted(expansion_words):
            weights.setdefault(word, .65)

        lexical_scores = self.lexical.search(weights)
        fetch = self.candidates * 3
        lexical = [doc_id for doc_id in sorted(lexical_scores, key=lexical_scores.get, reverse=True)
                   if admissible(self.documents[doc_id], context)][:fetch]
        dense_results = self.store.similarity_search(context.scenario, k=fetch)
        dense = [doc.id or doc.metadata.get("chunk_id") for doc in dense_results
                 if admissible(doc, context)]
        fused = reciprocal_ranks(lexical, dense)
        for doc_id in set(fused) - self.documents.keys():
            del fused[doc_id]

        def exact(doc):
            candidates = {context.reference_act} if context.reference_act else context.acts
            return context.reference == doc.metadata.get("section") and self.by_act[doc.id] in candidates

        if context.reference:
            for doc in self.documents.values():
                if exact(doc) and admissible(doc, context):
                    fused[doc.id] += .10
        if context.acts:
            for doc_id in fused:
                if self.by_act[doc_id] in context.acts:
                    fused[doc_id] += .035
        for doc_id in fused:
            if self.documents[doc_id].metadata.get("authority_type") == "primary_legislation":
                fused[doc_id] += .008
        ranked = sorted(fused, key=fused.get, reverse=True)[:self.candidates]
        if ranked and self.reranker is not None:
            pairs = [(context.scenario, self.rerank_text(self.documents[doc_id])) for doc_id in ranked]
            scores = self.reranker.predict(pairs, batch_size=8, show_progress_bar=False)
            rrf_order = {doc_id: index for index, doc_id in enumerate(ranked)}
            rerank_scores = dict(zip(ranked, scores))
            ranked.sort(key=lambda doc_id: (
                exact(self.documents[doc_id]) if context.reference else False,
                (self.by_act[doc_id] in context.acts) if context.acts else False,
                float(rerank_scores[doc_id]) - .12 * rrf_order[doc_id],
            ), reverse=True)

        matched = []
        for doc_id in ranked:
            doc = self.documents[doc_id]
            words = self.lexical.words[doc_id]
            authority_match = self.by_act[doc_id] in context.acts
            if context.acts and not authority_match:
                continue
            if context.reference and not exact(doc):
                continue
            text_overlap = words & (anchor_words | expansion_words)
            if exact(doc) or len(text_overlap - {"act", "section", "law", "court"}) >= (2 if authority_match else 3):
                matched.append(doc)

        if not matched:
            return Evidence(context, "insufficient", [], "I couldn't find reliable supporting material for this question.")
        matched = self.focus_issue(matched, context)
        selected = self.adjacent(matched, context, weights)
        primary = any(doc.metadata.get("authority_type") == "primary_legislation" for doc in selected)
        return Evidence(context, "relevant" if primary else "partial", selected,
                        None if primary else "I found background material, but no matching primary legal authority.")

    def focus_issue(self, matched, context):
        if context.reference or context.mode == "comparison":
            return matched
        for act, intent, issue in ISSUE_FOCUS:
            if act == "CrPC" and context.mode != "historical":
                continue
            if act in {"CPA", "RTI"} and act not in context.acts:
                continue
            if not re.search(intent, context.scenario, re.I):
                continue
            anchors = [
                doc
                for doc in matched
                if self.by_act[doc.id] == act and doc.metadata.get("section")
                and re.search(issue, " ".join(
                    (doc.metadata.get("section_heading", "") + " " + doc.page_content).split()
                ), re.I)
            ]
            provisions = {(doc.metadata["document"], doc.metadata["section"]) for doc in anchors}
            if len(provisions) == 1:
                pages = {doc.metadata["pdf_page"] for doc in anchors}
                selected = [doc for doc in matched if
                            (doc.metadata["document"], doc.metadata.get("section")) in provisions
                            and (act != "BNSS" or doc.metadata["pdf_page"] in pages)]
                anchor_ids = {doc.id for doc in anchors}
                return sorted(selected, key=lambda doc: doc.id in anchor_ids, reverse=True)
        return matched

    def adjacent(self, ranked, context, weights):
        selected = [ranked[0]]
        first = ranked[0]
        meta = first.metadata
        neighbors = self.by_provision.get((meta["document"], meta.get("section")), ())
        if self.by_act[first.id] == "BNSS" and re.search(FIR_REFUSAL, context.scenario, re.I):
            complete = complete_provision(
                (doc for doc in neighbors if doc.metadata["pdf_page"] == meta["pdf_page"] and admissible(doc, context)),
                r"\(4\)\s+Any person aggrieved by a refusal", r"make an application to the Magistrate\.", "4",
            )
            if complete:
                return [complete]
        if (self.by_act[first.id] == "BNS" and meta.get("section") == "303"
                and context.subsection in {None, "2"} and re.search(THEFT_PENALTY, context.scenario, re.I)):
            complete = complete_provision(
                (doc for doc in neighbors if admissible(doc, context)),
                r"\(2\)\s+Whoever commits theft shall be punished",
                r"shall be punished with community service\.", "2",
            )
            if complete:
                selected = [complete] + [doc for doc in selected if doc.id != complete.id]
        order = meta.get("document_order")
        if order is not None and self.top_k > 1:
            candidates = [doc for doc in neighbors if doc.id != first.id and
                          abs(doc.metadata.get("document_order", -10) - order) <= 6 and admissible(doc, context)]
            if (self.by_act[first.id] == "BNSS" and re.search(FIR_REFUSAL, context.scenario, re.I)
                    and re.search(FIR_ESCALATION, " ".join(first.page_content.split()), re.I)):
                candidates = [doc for doc in candidates if doc.metadata["pdf_page"] >= meta["pdf_page"]]
            if candidates:
                best = max(candidates, key=lambda doc: sum(
                    weights.get(word, 0) for word in self.lexical.words[doc.id]
                ))
                if any(weights.get(word, 0) for word in self.lexical.words[best.id] - self.lexical.words[first.id]):
                    selected.append(best)
        ids = {doc.id for doc in selected}
        return (selected + [doc for doc in ranked if doc.id not in ids])[:self.top_k]

    @staticmethod
    def rerank_text(doc):
        meta = doc.metadata
        label = " ".join(str(meta.get(key, "")) for key in ("act", "section", "section_heading"))
        return f"{label}\n{doc.page_content}"
