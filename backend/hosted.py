"""Hosted Gemini provider with bounded requests and a per-process call budget."""

import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic, sleep

import httpx
from google import genai
from google.genai import errors, types

from backend.rag_pipeline import (
    ANSWER_FORMAT, GENERAL_FORMAT, GENERAL_SYSTEM, PARTIAL_ANSWER_FORMAT, GenerationTimeout, ProviderUnavailable,
)

logger = logging.getLogger(__name__)
FREE_MODELS = frozenset({"gemini-3.1-flash-lite", "gemini-3.5-flash-lite"})
DEFAULT_MODEL = "gemini-3.1-flash-lite"
GROUNDED_INSTRUCTIONS = (
    "\nGive each required legal condition and exception its own supported claim. "
    "For procedural questions, give supported steps in order. Use plain English."
)


class RateLimited(ProviderUnavailable):
    pass


class Busy(ProviderUnavailable):
    pass


class GeminiNotConfigured(ProviderUnavailable):
    pass


class GeminiAuthenticationFailed(ProviderUnavailable):
    pass


class GeminiModelUnavailable(ProviderUnavailable):
    pass


def api_failure_reason(error):
    if error.code == 400 and isinstance(error.details, dict):
        body = error.details.get("error")
        details = body.get("details") if isinstance(body, dict) else None
        if isinstance(details, list) and any(
            isinstance(detail, dict) and detail.get("reason") == "API_KEY_INVALID" for detail in details
        ):
            return "authentication"
    return {400: "request_rejected", 401: "authentication", 403: "permission",
            404: "model_or_endpoint_unavailable", 408: "timeout", 429: "quota",
            503: "service_unavailable", 504: "timeout"}.get(error.code, "api_error")


class RequestBudget:
    def __init__(self, *limits, clock=monotonic):
        self.limits = limits
        self.history = deque()
        self.clock = clock
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = self.clock()
            longest = max(window for window, _ in self.limits)
            while self.history and self.history[0] <= now - longest:
                self.history.popleft()
            if any(sum(t > now - window for t in self.history) >= count for window, count in self.limits):
                raise RateLimited("Usage limit reached.")
            self.history.append(now)


@dataclass(frozen=True)
class HostedSettings:
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    deadline_sec: int = 36
    attempts: int = 2
    requests_per_minute: int = 4
    requests_per_day: int = 40
    output_tokens: int = 1536

    def __post_init__(self):
        if self.model not in FREE_MODELS:
            raise ValueError("Configure a supported free-tier Gemini model ID.")
        if (not 5 <= self.deadline_sec <= 60 or self.attempts not in {1, 2}
                or not 1 <= self.requests_per_minute <= 30 or not 1 <= self.requests_per_day <= 500
                or not 512 <= self.output_tokens <= 2048):
            raise ValueError("Hosted generation limits are outside the supported range.")


class Gemini:
    def __init__(self, settings, *, client=None, clock=monotonic, pause=sleep):
        self.settings = settings
        self.clock = clock
        self.pause = pause
        self.budget = RequestBudget((60, settings.requests_per_minute), (86400, settings.requests_per_day), clock=clock)
        self.client = client
        if self.client is None and settings.api_key:
            self.client = genai.Client(
                api_key=settings.api_key, vertexai=False,
                http_options=types.HttpOptions(api_version="v1beta", timeout=settings.deadline_sec * 1000,
                                               retry_options=types.HttpRetryOptions(attempts=1)),
            )

    def generate(self, messages):
        return self._generate(messages, ANSWER_FORMAT, GROUNDED_INSTRUCTIONS, self.settings.output_tokens)

    def generate_with_general(self, messages):
        instructions = (
            GROUNDED_INSTRUCTIONS + "\nUse the evidence-only rules for all claims. "
            "Only when incomplete is true, you may add a separate general object for the unsupported part. "
            "Omit general when the question is fully supported. Never move unsupported content into claims. "
            "The following rules apply only to that general object:\n" + GENERAL_SYSTEM
        )
        return self._generate(messages, PARTIAL_ANSWER_FORMAT, instructions, self.settings.output_tokens)

    def generate_general(self, messages):
        return self._generate(messages, GENERAL_FORMAT, "", 384)

    def _generate(self, messages, schema, instructions, output_tokens):
        if self.client is None:
            logger.warning("Gemini request skipped model=%s reason=missing_credentials; configure "
                           "[legalbot].api_key or GEMINI_API_KEY", self.settings.model)
            raise GeminiNotConfigured("Hosted generation has not been configured.")
        deadline = self.clock() + self.settings.deadline_sec
        for attempt in range(self.settings.attempts):
            remaining = deadline - self.clock()
            if remaining < 1:
                raise GenerationTimeout("Hosted generation timed out.")
            self.budget.acquire()
            try:
                response = self.client.models.generate_content(
                    model=self.settings.model, contents=messages[1]["content"],
                    config=types.GenerateContentConfig(
                        system_instruction=messages[0]["content"] + instructions,
                        response_mime_type="application/json", response_json_schema=schema,
                        temperature=0.15, max_output_tokens=output_tokens,
                        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                        http_options=types.HttpOptions(timeout=int(remaining * 1000),
                                                       retry_options=types.HttpRetryOptions(attempts=1)),
                    ),
                )
            except (httpx.TimeoutException, TimeoutError) as exc:
                logger.warning("Gemini request failed model=%s reason=timeout exception=%s",
                               self.settings.model, type(exc).__name__)
                raise GenerationTimeout("Hosted generation timed out.") from None
            except errors.APIError as exc:
                # Provider error bodies and exception strings may include confidential request data.
                reason = api_failure_reason(exc)
                logger.warning("Gemini request failed model=%s reason=%s exception=%s http_status=%s attempt=%d",
                               self.settings.model, reason, type(exc).__name__, exc.code, attempt + 1)
                if reason in {"authentication", "permission"}:
                    raise GeminiAuthenticationFailed("Gemini rejected the app's credentials or permissions.") from None
                if exc.code == 404:
                    raise GeminiModelUnavailable("The configured Gemini model is unavailable.") from None
                if exc.code == 429:
                    raise RateLimited("Hosted generation quota is temporarily unavailable.") from None
                if exc.code in {408, 504}:
                    raise GenerationTimeout("Hosted generation timed out.") from None
                if exc.code != 503 or attempt + 1 == self.settings.attempts or deadline - self.clock() < 1.5:
                    raise ProviderUnavailable("Hosted generation is temporarily unavailable.") from None
                self.pause(0.5)
                continue
            except httpx.TransportError as exc:
                logger.warning("Gemini request failed model=%s reason=transport exception=%s",
                               self.settings.model, type(exc).__name__)
                raise ProviderUnavailable("Hosted generation is temporarily unavailable.") from None
            text = response.text
            if not text:
                # Empty or safety-blocked output goes through the verified excerpt path.
                logger.warning("Gemini returned no answer text model=%s", self.settings.model)
                return "{}", self.settings.model
            return text, self.settings.model
        raise ProviderUnavailable("Hosted generation is temporarily unavailable.")
