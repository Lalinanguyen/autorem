"""The only gateway to the LLM. Every call goes through LLMClient.complete and is charged to the Budget.

Configured only through env vars: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. No
provider-specific code: any OpenAI-compatible /v1/chat/completions endpoint works.
Token counts come from the response's `usage` field. A response without usage is an
error, because the matched-budget comparison must never rest on estimates.

This is the only module that may import `openai` (tests/test_llm_client.py enforces it).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from harness.budget import Budget, Role

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class TransientLLMError(Exception):
    """Retryable failure: 429, 5xx, connection error, timeout."""


class MissingUsageError(Exception):
    pass


class Transport(Protocol):
    """Raw chat-completion call. Must raise TransientLLMError for retryable failures."""

    def __call__(self, messages: list[Message], max_tokens: int) -> LLMResponse: ...


class OpenAICompatTransport:
    def __init__(self, base_url: str, api_key: str, model: str, request_timeout_s: float = 600.0):
        import openai

        self._openai = openai
        # max_retries=0: this module owns retry policy so every attempt is visible to us.
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0, timeout=request_timeout_s
        )
        self.model = model

    def __call__(self, messages: list[Message], max_tokens: int) -> LLMResponse:
        oa = self._openai
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens
            )
        except (oa.RateLimitError, oa.APIConnectionError, oa.APITimeoutError) as e:
            raise TransientLLMError(str(e)) from e
        except oa.APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                raise TransientLLMError(f"HTTP {e.status_code}: {e}") from e
            raise
        if resp.usage is None or resp.usage.prompt_tokens is None or resp.usage.completion_tokens is None:
            raise MissingUsageError("endpoint returned no usage field; cannot charge budget")
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        return LLMResponse(text, int(resp.usage.prompt_tokens), int(resp.usage.completion_tokens))


class LLMClient:
    def __init__(
        self,
        budget: Budget,
        transport: Transport,
        *,
        max_retries: int = 6,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(budget, Budget):
            raise TypeError("LLMClient requires a Budget")
        self._budget = budget
        self._transport = transport
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep_fn

    @classmethod
    def from_env(cls, budget: Budget, **kwargs) -> "LLMClient":
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"missing env vars: {', '.join(missing)} (see .env.example)")
        transport = OpenAICompatTransport(
            os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"]
        )
        return cls(budget, transport, **kwargs)

    @property
    def budget(self) -> Budget:
        return self._budget

    def complete(
        self, messages: list[Message], *, role: Literal["agent", "sleep"], max_tokens: int
    ) -> LLMResponse:
        """One chat completion, charged to the budget under `role`.

        Raises BudgetExhausted before the call if nothing remains, or after it if the
        call pushed usage past the total (the response is then discarded).
        """
        max_tokens = self._budget.clamp_completion(max_tokens)
        attempt = 0
        while True:
            try:
                resp = self._transport(messages, max_tokens)
                break
            except TransientLLMError:
                if attempt >= self.max_retries:
                    raise
                delay = min(self.backoff_max_s, self.backoff_base_s * 2**attempt)
                self._sleep(delay * (0.5 + random.random() / 2))
                attempt += 1
        self._budget.charge(role, resp.prompt_tokens, resp.completion_tokens)
        return resp
