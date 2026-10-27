"""
llm_provider.py

Small provider abstraction for RAG answer generation.

Why this file exists:
  - rag_answer.py should not care whether the answer comes from Gemini, Groq,
    or a future provider.
  - fallback should happen for technical provider failures only: quota/rate
    limits, timeouts, temporary outages, missing primary key, etc.
  - retrieval quality is handled before this layer; switching LLMs cannot fix
    bad retrieved context.

Environment variables:
  LLM_PRIMARY_PROVIDER=gemini
  LLM_FALLBACK_PROVIDER=groq
  GEMINI_API_KEY=<set in .env or terminal>
  GEMINI_MODEL=gemini-3.5-flash
  GEMINI_THINKING_LEVEL=low
  GROQ_API_KEY=<set in .env or terminal>
  GROQ_MODEL=openai/gpt-oss-120b
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import threading
from typing import Protocol


DEFAULT_PRIMARY_PROVIDER = "gemini"
DEFAULT_FALLBACK_PROVIDER = "groq"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_THINKING_LEVEL = "low"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_LLM_TIMEOUT_SECONDS = 60


def load_env_file(filename: str = ".env") -> None:
    """
    Load simple KEY=value pairs from a .env file if present.

    This intentionally stays tiny: no dependency, no advanced shell parsing.
    Existing terminal environment variables win over .env values.
    """
    current_dir = Path(__file__).resolve().parent
    candidate_dirs = [current_dir, *current_dir.parents]

    env_path = None
    for directory in candidate_dirs:
        candidate = directory / filename
        if candidate.exists():
            env_path = candidate
            break

    if env_path is None:
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


class LLMProviderError(Exception):
    """
    Provider-level error with a retry/fallback signal.

    retryable=True means another provider may reasonably be tried.
    retryable=False means the request itself is invalid or all configured
    credentials/dependencies are missing.
    """

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    fallback_used: bool
    primary_error: str | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        ...


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.thinking_level = os.getenv(
            "GEMINI_THINKING_LEVEL",
            DEFAULT_GEMINI_THINKING_LEVEL,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self.api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY is not set.",
                retryable=True,
            )

        try:
            from google import genai
        except ImportError as exc:
            raise LLMProviderError(
                "google-genai is not installed. Run: pip install google-genai",
                retryable=True,
            ) from exc

        try:
            client = genai.Client(api_key=self.api_key)
            interaction = client.interactions.create(
                model=self.model,
                system_instruction=system_prompt,
                input=user_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "thinking_level": self.thinking_level,
                },
            )
            text = getattr(interaction, "output_text", "") or ""
            if not text.strip():
                raise LLMProviderError(
                    "Gemini returned an empty response.",
                    retryable=True,
                )
            return text.strip()
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"Gemini request failed: {exc}",
                retryable=True,
            ) from exc


class GroqProvider:
    name = "groq"

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self.api_key:
            raise LLMProviderError(
                "GROQ_API_KEY is not set.",
                retryable=True,
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMProviderError(
                "groq is not installed. Run: pip install groq",
                retryable=True,
            ) from exc

        try:
            client = Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
            )
            text = completion.choices[0].message.content or ""
            if not text.strip():
                raise LLMProviderError(
                    "Groq returned an empty response.",
                    retryable=True,
                )
            return text.strip()
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"Groq request failed: {exc}",
                retryable=True,
            ) from exc


def _provider_from_name(name: str) -> LLMProvider:
    normalized = name.strip().lower()
    if normalized == "gemini":
        return GeminiProvider()
    if normalized == "groq":
        return GroqProvider()
    raise LLMProviderError(
        f"Unsupported LLM provider: {name}. Use 'gemini' or 'groq'.",
        retryable=False,
    )


def configured_providers() -> list[LLMProvider]:
    """
    Return providers in the configured fallback order.
    """
    primary = os.getenv("LLM_PRIMARY_PROVIDER", DEFAULT_PRIMARY_PROVIDER)
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", DEFAULT_FALLBACK_PROVIDER)

    provider_names = []
    for name in [primary, fallback]:
        normalized = name.strip().lower()
        if normalized and normalized not in provider_names:
            provider_names.append(normalized)

    return [_provider_from_name(name) for name in provider_names]


def _short_error(message: str) -> str:
    """
    Keep CLI fallback errors readable while preserving the useful reason.
    """
    lower = message.lower()
    if "api_key_invalid" in lower or "api key not valid" in lower:
        return "API key invalid"
    if "quota" in lower or "rate limit" in lower or "429" in lower:
        return "quota/rate limit reached"
    if "timeout" in lower or "timed out" in lower:
        return "request timed out"
    if "not set" in lower or "not installed" in lower:
        return message
    return message.splitlines()[0][:240]


def _provider_timeout_seconds() -> int:
    raw_value = os.getenv("LLM_TIMEOUT_SECONDS")
    if not raw_value:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        return max(5, int(raw_value))
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS


def _call_provider_with_timeout(
    provider: LLMProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: int,
) -> str:
    """
    Run a provider call in a daemon thread so a slow API request cannot hang
    the CLI forever.
    """
    result_queue: queue.Queue[tuple[str, str | BaseException]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            text = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            result_queue.put(("ok", text))
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise LLMProviderError(
            f"{provider.name} request timed out after {timeout_seconds} seconds.",
            retryable=True,
        )

    status, payload = result_queue.get()
    if status == "ok":
        return str(payload)
    raise payload


def generate_with_fallback(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
    timeout_seconds: int | None = None,
) -> LLMResponse:
    """
    Try configured providers in order and return the first successful answer.
    """
    providers = configured_providers()
    if not providers:
        raise LLMProviderError("No LLM providers configured.", retryable=False)

    first_error: str | None = None
    errors = []
    timeout_seconds = timeout_seconds or _provider_timeout_seconds()

    for index, provider in enumerate(providers):
        try:
            text = _call_provider_with_timeout(
                provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )
            return LLMResponse(
                text=text,
                provider=provider.name,
                model=provider.model,
                fallback_used=index > 0,
                primary_error=first_error if index > 0 else None,
            )
        except LLMProviderError as exc:
            error_text = f"{provider.name}: {_short_error(str(exc))}"
            errors.append(error_text)
            if first_error is None:
                first_error = error_text
            if not exc.retryable:
                break

    raise LLMProviderError(
        "All configured LLM providers failed. " + " | ".join(errors),
        retryable=False,
    )
