"""
summarizer.py
~~~~~~~~~~~~~
LLM summarization component with prompt versioning and provider abstractions (Phase 7).

Follows Roadmap Section 6, 7.7, and 9.1:
  - Abstract interface BaseSummarizer.
  - GeminiSummarizer backed by the official `google-genai` SDK.
  - MockSummarizer for zero-cost, network-free automated testing.
  - Versioned summarization prompt templates (e.g. 'v1.0').
  - Temperature 0.0 with concise, faithful output instructions.
  - Clean error handling that never exposes API keys or secrets in logs/messages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Optional

from research_monitor import config

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_VERSION = "v1.0"

PROMPT_TEMPLATES: dict[str, str] = {
    "v1.0": (
        "You are an expert AI research assistant. Your task is to write a concise, faithful summary "
        "of an academic paper framed around the user's specific research interests.\n\n"
        "User Research Interests:\n"
        "{user_interests}\n\n"
        "Paper Title:\n"
        "{title}\n\n"
        "Paper Abstract:\n"
        "{abstract}\n\n"
        "Instructions:\n"
        "1. Write a 2-3 sentence summary explaining the core method/contribution and directly "
        "connecting it to the user's research interests.\n"
        "2. Remain strictly faithful to the paper abstract. Never invent methods, results, claims, "
        "or findings absent from the source.\n"
        "3. Do not include markdown headers, bullet points, or meta-chatter. Provide only the interest-framed summary text."
    ),
    "v2.0": (
        "You are an AI research assistant. Summarize the following paper specifically focusing on "
        "how it addresses the user's research goals.\n\n"
        "User Interests: {user_interests}\n"
        "Title: {title}\n"
        "Abstract: {abstract}\n\n"
        "Instructions:\n"
        "- Exactly two sentences.\n"
        "- Sentence 1: Key finding or technique.\n"
        "- Sentence 2: Why it matters for the specified interests.\n"
        "- Strictly no hallucinated claims."
    ),
}


@dataclass
class SummaryResult:
    """Standardized result returned by all summarizer implementations."""
    text: str
    model: str
    prompt_version: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


class BaseSummarizer(ABC):
    """Abstract interface for LLM paper summarization."""

    @abstractmethod
    def generate_summary(
        self,
        title: str,
        abstract: str,
        user_interests: str,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> SummaryResult:
        """
        Generate an interest-framed summary from title, abstract, and user interests.

        Parameters
        ----------
        title : str
            Title of the paper.
        abstract : str
            Source abstract of the paper.
        user_interests : str
            User research interests description or seed query.
        prompt_version : str
            Version identifier of the prompt template (default: 'v1.0').

        Returns
        -------
        SummaryResult
            Containing generated text, model identifier, prompt version, and token counts.
        """
        pass

    def render_prompt(
        self,
        title: str,
        abstract: str,
        user_interests: str,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> str:
        """Format the prompt template for the given version."""
        template = PROMPT_TEMPLATES.get(prompt_version)
        if not template:
            raise ValueError(f"Unknown summary prompt version: '{prompt_version}'")
        return template.format(
            title=title.strip(),
            abstract=abstract.strip(),
            user_interests=user_interests.strip() if user_interests else "General artificial intelligence and machine learning research",
        )


class GeminiSummarizer(BaseSummarizer):
    """
    Hosted Google Gemini summarizer using the official `google-genai` SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self._client: Optional[Any] = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY environment variable.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # Never expose API key or sensitive details
                logger.error("Failed to initialize Google GenAI client: %s", type(e).__name__)
                raise RuntimeError("Could not initialize LLM client.") from e
        return self._client

    def generate_summary(
        self,
        title: str,
        abstract: str,
        user_interests: str,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> SummaryResult:
        prompt = self.render_prompt(
            title=title,
            abstract=abstract,
            user_interests=user_interests,
            prompt_version=prompt_version,
        )

        client = self._get_client()

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                ),
            )

            text = response.text.strip() if response.text else ""
            if not text:
                raise RuntimeError("LLM returned an empty summary response.")

            tokens_in = None
            tokens_out = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_in = getattr(response.usage_metadata, "prompt_token_count", None)
                tokens_out = getattr(response.usage_metadata, "candidates_token_count", None)

            return SummaryResult(
                text=text,
                model=self.model,
                prompt_version=prompt_version,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        except Exception as e:
            # Clean logging: never log raw credentials or full trace with secrets
            logger.error(
                "Gemini summarization failed for model '%s', prompt '%s': %s: %s",
                self.model,
                prompt_version,
                type(e).__name__,
                str(e),
            )
            raise


class MockSummarizer(BaseSummarizer):
    """
    Deterministic mock summarizer for automated tests and offline verification.
    """

    def __init__(
        self,
        model: str = "mock-gemini-flash",
        custom_summary: Optional[str] = None,
        should_fail: bool = False,
        failure_exception: Optional[Exception] = None,
        fail_times: int = 0,
    ):
        self.model = model
        self.custom_summary = custom_summary
        self.should_fail = should_fail
        self.failure_exception = failure_exception
        self.fail_times = fail_times
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def generate_summary(
        self,
        title: str,
        abstract: str,
        user_interests: str,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> SummaryResult:
        prompt = self.render_prompt(
            title=title,
            abstract=abstract,
            user_interests=user_interests,
            prompt_version=prompt_version,
        )

        self.call_count += 1
        self.calls.append({
            "title": title,
            "abstract": abstract,
            "user_interests": user_interests,
            "prompt_version": prompt_version,
            "prompt": prompt,
        })

        if self.should_fail or (self.call_count <= self.fail_times):
            exc = self.failure_exception or RuntimeError("Simulated LLM generation failure.")
            raise exc

        if self.custom_summary:
            summary_text = self.custom_summary
        else:
            # Deterministic, interest-framed summary
            short_abs = abstract[:140].strip() if abstract else ""
            summary_text = (
                f"This work on '{title}' directly aligns with your interest in {user_interests}. "
                f"It proposes novel methods to address key challenges ({short_abs})."
            )

        # Approximate token counts
        tokens_in = max(1, len(prompt) // 4)
        tokens_out = max(1, len(summary_text) // 4)

        return SummaryResult(
            text=summary_text,
            model=self.model,
            prompt_version=prompt_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def get_summarizer(
    provider: Optional[str] = None,
    mock: bool = False,
    **kwargs: Any,
) -> BaseSummarizer:
    """
    Factory to retrieve a configured summarizer instance.
    If mock=True or GEMINI_API_KEY is not configured, returns a MockSummarizer.
    """
    if mock:
        return MockSummarizer(**kwargs)

    # If provider explicitly requested as mock or no API key configured
    if provider == "mock" or not config.GEMINI_API_KEY:
        return MockSummarizer(**kwargs)

    return GeminiSummarizer(
        api_key=config.GEMINI_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        **kwargs,
    )
