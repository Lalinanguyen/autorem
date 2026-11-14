"""The only gateway to the LLM. Every call goes through LLMClient.complete and is charged to the Budget.

Endpoint configured only through env vars: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. No
provider-specific code: any OpenAI-compatible /v1/chat/completions endpoint works (OpenAI, a
vLLM server on Modal, ...). Model-dependent request parameters (max_completion_tokens vs
max_tokens, reasoning_effort, temperature) are resolved once per run by llm/chat_params.py from
RunConfig.llm, env vars (LLM_TOKEN_PARAM, LLM_REASONING_EFFORT, LLM_TEMPERATURE,
LLM_REASONING_MODEL) and auto-detection by model name, and are identical for agent and sleep calls.

Token counts come from the response's `usage` field. A response without usage is an error,
because the matched-budget comparison must never rest on estimates. The Budget charges
prompt_tokens + completion_tokens; cached prompt tokens and reasoning tokens (subsets of those
two) are recorded alongside for logging and cost.

Billing / quota errors (llm/chat_params.py BILLING_ERROR_CODES, e.g. HTTP 429 insufficient_quota)
raise BillingError immediately: they are never retried.

This is the only module that may import `openai` (tests.py enforces it).
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from harness.budget import Budget, Role
from llm.chat_params import BillingError, ChatParams, is_billing_error, parse_usage, resolve_chat_params

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

__all__ = ["BillingError", "ChatParams", "LLMClient", "LLMResponse", "MissingUsageError", "OpenAICompatTransport",
           "TransientLLMError", "strip_think"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop <think>...</think> blocks (Qwen chain-of-thought on the vLLM path).

    With enable_thinking on, vLLM returns reasoning inside <think> tags in `content`
    (or in a separate `reasoning_content` field, which we ignore — it's already out of
    `content`). Stripping keeps downstream parsers (code extraction, consolidator JSON)
    from ever seeing the reasoning trace. ChatParams.thinking is the on/off switch; this
    function is unconditional and harmless when no tags are present."""
    if "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None      # usage.prompt_tokens_details.cached_tokens (subset of prompt)
    reasoning_tokens: int | None = None   # usage.completion_tokens_details.reasoning_tokens (subset of completion)


class TransientLLMError(Exception):
    """Retryable failure: 429 (rate limit, not billing), 5xx, connection error, timeout."""


class MissingUsageError(Exception):
    pass


class Transport(Protocol):
    """Raw chat-completion call. Must raise TransientLLMError for retryable failures and
    BillingError for billing / quota refusals."""

    def __call__(self, messages: list[Message], max_tokens: int) -> LLMResponse: ...


def _billing(e: Any) -> BillingError | None:
    code, type_ = getattr(e, "code", None), getattr(e, "type", None)
    body = getattr(e, "body", None)
    if isinstance(body, dict):  # the SDK sets body to the inner error object
        code, type_ = code or body.get("code"), type_ or body.get("type")
    if is_billing_error(code, type_):
        return BillingError(str(e)[:500], status=getattr(e, "status_code", None),
                            code=code if is_billing_error(code) else type_)
    return None


class OpenAICompatTransport:
    def __init__(self, base_url: str, api_key: str, model: str, request_timeout_s: float = 600.0,
                 params: ChatParams | None = None):
        import openai

        self._openai = openai
        # max_retries=0: this module owns retry policy so every attempt is visible to us.
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0, timeout=request_timeout_s
        )
        self.model = model
        self.params = params or resolve_chat_params(model, env={}, warn=False)

    def __call__(self, messages: list[Message], max_tokens: int) -> LLMResponse:
        oa = self._openai
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, **self.params.request_kwargs(max_tokens)
            )
        except oa.APIStatusError as e:
            billing = _billing(e)
            if billing is not None:
                raise billing from e
            if e.status_code == 429 or e.status_code >= 500:
                raise TransientLLMError(f"HTTP {e.status_code}: {e}") from e
            raise
        except (oa.APIConnectionError, oa.APITimeoutError) as e:
            raise TransientLLMError(str(e)) from e
        usage = parse_usage(resp.usage)
        if usage is None:
            raise MissingUsageError("endpoint returned no usage field; cannot charge budget")
        text = strip_think((resp.choices[0].message.content or "") if resp.choices else "")
        return LLMResponse(text, usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens,
                           usage.reasoning_tokens)


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
    def from_env(cls, budget: Budget, settings: Any = None, **kwargs) -> "LLMClient":
        """Client on LLM_BASE_URL / LLM_API_KEY / LLM_MODEL; `settings` = RunConfig.llm (or None)."""
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"missing env vars: {', '.join(missing)} (see .env.example)")
        model = os.environ["LLM_MODEL"]
        transport = OpenAICompatTransport(
            os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], model,
            params=resolve_chat_params(model, settings),
        )
        return cls(budget, transport, **kwargs)

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def chat_params(self) -> ChatParams | None:
        """The resolved request parameters (None for transports without them, e.g. the mock)."""
        return getattr(self._transport, "params", None)

    @property
    def model(self) -> str | None:
        return getattr(self._transport, "model", None)

    def complete(
        self, messages: list[Message], *, role: Literal["agent", "sleep"], max_tokens: int
    ) -> LLMResponse:
        """One chat completion, charged to the budget under `role`.

        Raises BudgetExhausted before the call if nothing remains, or after it if the
        call pushed usage past the total (the response is then discarded). Raises
        BillingError (never retried) on a billing / quota refusal.
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
        self._budget.charge(role, resp.prompt_tokens, resp.completion_tokens,
                            cached_tokens=getattr(resp, "cached_tokens", None),
                            reasoning_tokens=getattr(resp, "reasoning_tokens", None))
        return resp
