"""Runtime configuration for the local LegalBot services."""

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


def _number(name, default, minimum, maximum, convert=int):
    value = convert(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b-instruct-q4_K_M"
    fallback_model: str = "qwen2.5:3b-instruct-q4_K_M"
    cloud_model: str = ""
    rerank: bool = False
    candidates: int = 18
    top_k: int = 5
    threads: int = 4
    context_tokens: int = 4096
    output_tokens: int = 512
    temperature: float = .15
    timeout: float = 240
    keep_alive: str = "5m"

    @classmethod
    def from_env(cls):
        url = os.getenv("LEGALBOT_OLLAMA_URL", cls.ollama_url).rstrip("/")
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("LEGALBOT_OLLAMA_URL must be a loopback HTTP origin")
        candidates = _number("LEGALBOT_CANDIDATES", cls.candidates, 5, 50)
        return cls(
            ollama_url=url,
            model=os.getenv("LEGALBOT_MODEL", cls.model),
            fallback_model=os.getenv("LEGALBOT_FALLBACK_MODEL", cls.fallback_model),
            cloud_model=os.getenv("LEGALBOT_CLOUD_MODEL", "").strip(),
            rerank=os.getenv("LEGALBOT_RERANK", "0").lower() in {"1", "true"},
            candidates=candidates,
            top_k=_number("LEGALBOT_TOP_K", cls.top_k, 1, min(candidates, 8)),
            threads=_number("LEGALBOT_THREADS", cls.threads, 1, 16),
            context_tokens=_number("LEGALBOT_CONTEXT_TOKENS", cls.context_tokens, 3072, 8192),
            output_tokens=_number("LEGALBOT_OUTPUT_TOKENS", cls.output_tokens, 128, 1024),
            temperature=_number("LEGALBOT_TEMPERATURE", cls.temperature, 0, 1, float),
            timeout=_number("LEGALBOT_TIMEOUT", cls.timeout, 5, 900, float),
            keep_alive=os.getenv("LEGALBOT_KEEP_ALIVE", cls.keep_alive),
        )
