"""Shared resources for Streamlit's direct retrieval and hosted inference."""

import os
from dataclasses import dataclass, field
from threading import BoundedSemaphore
from time import perf_counter

from langchain_chroma import Chroma

from backend.db import COLLECTION_NAME, validate_collection
from backend.hosted import Busy, DEFAULT_MODEL, Gemini, HostedSettings, RequestBudget
from backend.index_bundle import prepare_bundle
from backend.rag_pipeline import LegalAssistant
from backend.retrieval import LegalRetriever
from backend.transition import generate_explanation, load_catalog
from config import Settings

MAX_QUESTION = 1600
MAX_HISTORY = 4


@dataclass(frozen=True)
class CloudSettings:
    provider: HostedSettings = field(default_factory=HostedSettings)
    index_url: str = ""
    index_sha256: str = ""
    index_bundle: str = ""
    rerank: bool = False

    @classmethod
    def from_mapping(cls, values):
        def read(name, env, default=""):
            return values.get(name, os.getenv(env, default))

        return cls(
            provider=HostedSettings(
                api_key=(str(values.get("api_key", "")).strip() or os.getenv("GEMINI_API_KEY", "").strip()
                         or os.getenv("GOOGLE_API_KEY", "").strip()),
                model=str(read("model", "LEGALBOT_GEMINI_MODEL", DEFAULT_MODEL)).strip(),
                requests_per_minute=int(read("requests_per_minute", "LEGALBOT_HOSTED_RPM", "4")),
                requests_per_day=int(read("requests_per_day", "LEGALBOT_HOSTED_RPD", "40")),
            ),
            index_url=str(read("index_url", "LEGALBOT_INDEX_URL")).strip(),
            index_sha256=str(read("index_sha256", "LEGALBOT_INDEX_SHA256")).strip().lower(),
            index_bundle=str(read("index_bundle", "LEGALBOT_INDEX_BUNDLE")).strip(),
            rerank=str(read("rerank", "LEGALBOT_RERANK", "false")).lower() in {"true", "1"},
        )


def create_retriever(config):
    from backend.onnx_models import MiniLM, OnnxReranker

    directory, count = prepare_bundle(config.index_sha256, local=config.index_bundle, url=config.index_url)
    client = validate_collection(directory, count)
    store = Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=MiniLM(directory),
                   create_collection_if_not_exists=False)
    return LegalRetriever(store, candidates=18, top_k=5, reranker=OnnxReranker(2) if config.rerank else None)


class DemoService:
    def __init__(self, config, *, provider=None, retriever_factory=None):
        self.config = config
        self.slot = BoundedSemaphore(1)
        self.assistant = LegalAssistant(
            Settings(), provider=provider if provider is not None else Gemini(config.provider),
            retriever_factory=retriever_factory or (lambda: create_retriever(config)),
        )

    def ask(self, question, history, context, session_budget=None):
        question = question.strip()
        if not question or len(question) > MAX_QUESTION:
            raise ValueError("Enter a question of at most 1,600 characters.")
        if not self.slot.acquire(blocking=False):
            raise Busy("The demo is answering another question.")
        try:
            if session_budget is not None:
                session_budget.acquire()
            start = perf_counter()
            history = [item[:MAX_QUESTION] for item in history[-MAX_HISTORY:] if isinstance(item, str)]
            result = self.assistant.answer(question, history=history, **context)
            result["response_time_sec"] = round(perf_counter() - start, 3)
            return result
        finally:
            self.slot.release()

    def explain_transition(self, mapping_id, session_budget):
        catalog = load_catalog()
        mapping = catalog.mappings[mapping_id]
        if not self.slot.acquire(blocking=False):
            raise Busy("The demo is answering another question.")
        try:
            session_budget.acquire()
            return generate_explanation(catalog, mapping, self.assistant.get_retriever, self.assistant.provider)
        finally:
            self.slot.release()


def session_gate():
    return RequestBudget((2, 1), (600, 10))
