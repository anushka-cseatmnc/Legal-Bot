"""FastAPI interface for the shared local legal assistant."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date
from functools import partial
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.db import EMBEDDING_MODEL, EMBEDDING_REVISION, PROJECT_ROOT, IndexUnavailable, index_root, read_index_manifest, validate_collection
from backend.rag_pipeline import GenerationTimeout, LegalAssistant, ProviderUnavailable
from backend.retrieval import RERANKER, RERANKER_REVISION
from config import Settings

logger = logging.getLogger(__name__)
EvidenceStatus = Literal["relevant", "partial", "insufficient", "out_of_scope", "error"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1600)
    history: list[str] = Field(default_factory=list, max_length=4)
    event_date: date | None = None
    jurisdiction: str | None = Field(default=None, max_length=24)
    procedural_status: Literal["pending_at_transition", "not_pending_at_transition", "unsure"] | None = None


class Citation(BaseModel):
    id: str
    document: str
    act: str | None = None
    section: str | None = None
    subsection: str | None = None
    section_heading: str | None = None
    provision_type: str | None = None
    pdf_page: int
    page_label: str
    authority_type: str
    legal_status: str
    source_url: str | None = None
    effective_from: str | None = None
    repealed_on: str | None = None
    savings: str | None = None
    commencement_source: str | None = None
    commencement_notification: str | None = None
    excerpt: str
    quotes: list[str]


class QueryResponse(BaseModel):
    answer: str
    response: str
    citations: list[Citation] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    status: EvidenceStatus
    response_time_sec: float
    legal_context: dict = Field(default_factory=dict)
    model: str | None = None
    answer_kind: Literal["generated", "excerpt", "metadata", "abstention"] = "abstention"
    error: dict | None = None


def check_index(settings):
    from huggingface_hub import hf_hub_download

    root = index_root()
    manifest = read_index_manifest(root)
    validate_collection(root / "snapshots" / manifest["snapshot"], manifest["chunks"])
    hf_hub_download(EMBEDDING_MODEL, "model.safetensors", revision=EMBEDDING_REVISION, local_files_only=True)
    if settings.rerank:
        hf_hub_download(RERANKER, "model.safetensors", revision=RERANKER_REVISION, local_files_only=True)


def create_app(settings=None, *, assistant=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        load_dotenv(PROJECT_ROOT / ".env")
        app.state.settings = settings or Settings.from_env()
        app.state.assistant = assistant or LegalAssistant(app.state.settings)
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legalbot")
        app.state.inflight = None
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.executor.shutdown, wait=True, cancel_futures=True)
            await asyncio.to_thread(app.state.assistant.provider.close)

    app = FastAPI(title="LegalBot", lifespan=lifespan)

    def error(status, code, message, started):
        body = QueryResponse(answer=message, response=message, status="error",
                             response_time_sec=round(time.perf_counter() - started, 3),
                             error={"code": code, "message": message})
        return JSONResponse(body.model_dump(), status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return error(422, "invalid_request", "Check the question and optional legal context fields.", time.perf_counter())

    @app.post("/chat", response_model=QueryResponse)
    @app.post("/query/", response_model=QueryResponse)
    async def query(req: QueryRequest, request: Request):
        started = time.perf_counter()
        question = req.question.strip()
        if not question:
            return error(422, "invalid_question", "Enter a non-empty legal question.", started)
        if sum(map(len, req.history)) > 6400 or any(len(item) > 1600 for item in req.history):
            return error(422, "invalid_history", "Conversation history is too long.", started)
        state = request.app.state
        if state.inflight is not None and not state.inflight.done():
            return error(429, "busy", "The local assistant is already processing a question.", started)
        call = partial(state.assistant.answer, question, history=req.history,
                       event_date=req.event_date.isoformat() if req.event_date else None,
                       jurisdiction=req.jurisdiction, procedural_status=req.procedural_status)
        future = asyncio.get_running_loop().run_in_executor(state.executor, call)
        state.inflight = future
        # Retrieve a late exception even after an HTTP timeout or client disconnect.
        future.add_done_callback(lambda completed: None if completed.cancelled() else completed.exception())
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=state.settings.timeout + 8)
        except (asyncio.TimeoutError, TimeoutError, GenerationTimeout):
            return error(504, "timeout", "The local assistant exceeded its time budget.", started)
        except (IndexUnavailable, ProviderUnavailable, ImportError, OSError):
            return error(503, "unavailable", "The legal index or configured model is unavailable. Check backend readiness.", started)
        except Exception as exc:
            logger.error("Legal request failed (%s)", type(exc).__name__)
            return error(500, "internal_error", "The assistant could not complete this request.", started)
        return QueryResponse(
            **result, response=result["answer"],
            source_documents=[source["excerpt"] for source in result["citations"]],
            response_time_sec=round(time.perf_counter() - started, 3),
        )

    @app.get("/")
    async def root():
        return {"message": "LegalBot. Send a question to /chat or /query/."}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        state = request.app.state
        try:
            await asyncio.to_thread(check_index, state.settings)
        except Exception as exc:
            logger.info("Index readiness check failed (%s)", type(exc).__name__)
            return JSONResponse({"status": "unavailable", "component": "index_or_embedding"}, status_code=503)
        if not await asyncio.to_thread(state.assistant.provider.ready):
            return JSONResponse({"status": "unavailable", "component": "generation"}, status_code=503)
        return {"status": "ready"}

    return app


app = create_app()
