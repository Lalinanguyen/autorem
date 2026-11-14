"""OpenAI-compatible budget proxy: lets third-party optimizers (GEPA) spend the run's Budget.

Code we do not control (e.g. gepa's reflection step) cannot call `LLMClient.complete`, but it
can talk to an OpenAI-compatible URL. This proxy is that URL. It forwards (stdlib urllib, no extra dependency)
`POST /v1/chat/completions` to `LLM_BASE_URL` (with `LLM_API_KEY`, model forced to
`LLM_MODEL`) and charges the upstream `usage` to the Budget registered for the request's
`X-Run-Id` header, with the same rules as `LLMClient` (harness/budget.py):

  1. Before forwarding: refuse if nothing remains, and clamp `max_tokens` /
     `max_completion_tokens` to the remaining budget (a missing value is set to
     min(default_max_tokens, remaining)).
  2. After the upstream answers: charge `usage.prompt_tokens + usage.completion_tokens`
     exactly as reported. A response without usage is an error (HTTP 502,
     code "missing_usage"); tokens are never estimated. If the charge pushes the run over
     its total, the tokens stay recorded and the response is withheld.

Request parameters: the proxy applies the run's resolved llm/chat_params.py ChatParams to every
forwarded body (the same resolution LLMClient uses: max_completion_tokens for reasoning models,
reasoning_effort, temperature / top_p / logprobs stripped where the model rejects them), and records
the usage breakdown (cached prompt tokens, reasoning tokens) with each charge.

Billing / quota refusals from the upstream (e.g. HTTP 429 insufficient_quota) are passed through
unchanged; `ProxyLM` raises BillingError for them without retrying.

Refusals use HTTP 402 with an OpenAI-style body
`{"error": {"code": "budget_exhausted", "type": "budget_exhausted", ...}}`. `ProxyLM` (the
client half, used by arms/) turns that into `BudgetExhausted`.

Headers: `X-Run-Id` (required), `X-Budget-Role` ("agent" default, or "sleep").
Requests for one run are serialized, so overshoot stays at most one call, as in LLMClient.
Streaming is refused (usage must be read from the final body).

Run standalone (budgets are in-process, so this is mainly for debugging):
    uv run python -m infra.llm_proxy --port 8765
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest

from harness.budget import Budget, BudgetExhausted
from llm.chat_params import BillingError, ChatParams, billing_error_from_body, parse_usage, resolve_chat_params

RUN_ID_HEADER = "X-Run-Id"
ROLE_HEADER = "X-Budget-Role"
BUDGET_EXHAUSTED = "budget_exhausted"
_COMPLETIONS_SUFFIX = "/chat/completions"
# Explicit User-Agent: some upstreams (Cloudflare in front of Tinker) refuse urllib's default
# "Python-urllib/3.x" with HTTP 403 "error code: 1010".
USER_AGENT = "sleep-evolve-proxy/0.1"


class _ConnectionFailed(Exception):
    pass


def post_json(url: str, body: dict, headers: dict[str, str], timeout_s: float) -> tuple[int, Any]:
    """POST JSON with the stdlib. Returns (status, parsed JSON or raw text). Raises _ConnectionFailed."""
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST", headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            status, raw = resp.status, resp.read()
    except urlerror.HTTPError as e:
        status, raw = e.code, e.read()
    except (urlerror.URLError, OSError) as e:  # refused, DNS, timeout
        raise _ConnectionFailed(f"{type(e).__name__}: {e}") from e
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode("utf-8", errors="replace")


class ProxyError(Exception):
    """Non-retryable error returned by the proxy or upstream (bad request, missing usage...)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status, self.code = status, code


@dataclass
class _Run:
    budget: Budget
    lock: threading.Lock = field(default_factory=threading.Lock)
    calls: list[dict] = field(default_factory=list)  # one record per upstream call that returned usage


class LLMProxy:
    """In-process HTTP proxy. `start()` binds 127.0.0.1:<port> (0 = any free port)."""

    def __init__(
        self,
        upstream_base_url: str,
        api_key: str = "",
        model: str | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        default_max_tokens: int = 4096,
        request_timeout_s: float = 600.0,
        chat_params: ChatParams | None = None,
    ):
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.chat_params = chat_params or resolve_chat_params(model, env={}, warn=False)
        self.default_max_tokens = default_max_tokens
        self._runs: dict[str, _Run] = {}
        self._runs_lock = threading.Lock()
        self.request_timeout_s = request_timeout_s
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, settings: Any = None, **kwargs: Any) -> "LLMProxy":
        """Proxy to LLM_BASE_URL / LLM_MODEL; `settings` = RunConfig.llm (or None: env, then auto)."""
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"missing env vars: {', '.join(missing)} (see .env.example)")
        model = os.environ["LLM_MODEL"]
        kwargs.setdefault("chat_params", resolve_chat_params(model, settings))
        return cls(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], model, **kwargs)

    # ------------------------------------------------------------------ lifecycle

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "LLMProxy":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True, name="llm-proxy"
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "LLMProxy":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ budgets

    def register(self, run_id: str, budget: Budget) -> None:
        if not isinstance(budget, Budget):
            raise TypeError("register() requires a Budget")
        with self._runs_lock:
            if run_id in self._runs and self._runs[run_id].budget is not budget:
                raise ValueError(f"run_id {run_id!r} is already registered with another Budget")
            self._runs.setdefault(run_id, _Run(budget))

    def unregister(self, run_id: str) -> None:
        with self._runs_lock:
            self._runs.pop(run_id, None)

    def usage_log(self, run_id: str) -> list[dict]:
        with self._runs_lock:
            run = self._runs.get(run_id)
        return list(run.calls) if run else []

    # ------------------------------------------------------------------ request handling

    def handle_completion(self, headers: Any, raw_body: bytes) -> tuple[int, dict]:
        """Core logic, independent of the HTTP server (returns status, JSON body)."""
        run_id = headers.get(RUN_ID_HEADER)
        if not run_id:
            return _err(400, "missing_run_id", f"the {RUN_ID_HEADER} header is required")
        with self._runs_lock:
            run = self._runs.get(run_id)
        if run is None:
            return _err(404, "unknown_run_id", f"no Budget registered for run_id {run_id!r}")
        role = headers.get(ROLE_HEADER) or "agent"
        if role not in ("agent", "sleep"):
            return _err(400, "bad_role", f"{ROLE_HEADER} must be 'agent' or 'sleep', got {role!r}")
        try:
            body = json.loads(raw_body or b"{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return _err(400, "bad_json", "request body must be a JSON object")
        if body.get("stream"):
            return _err(400, "streaming_unsupported", "stream=true is not supported: usage must be charged")

        budget = run.budget
        with run.lock:  # one in-flight call per run -> overshoot stays at most one call
            try:
                requested = body.get("max_completion_tokens") or body.get("max_tokens") or self.default_max_tokens
                key = self.chat_params.token_key_for(body)
                clamped = budget.clamp_completion(int(requested))
            except BudgetExhausted as e:
                return _err(402, BUDGET_EXHAUSTED, str(e), budget=budget.snapshot())
            self.chat_params.apply_to_body(body)
            body.pop("max_tokens", None)
            body.pop("max_completion_tokens", None)
            body[key] = clamped
            if self.model:
                body["model"] = self.model
            headers_out: dict[str, str] = {}
            if self.api_key:
                headers_out["Authorization"] = f"Bearer {self.api_key}"
            try:
                status, data = post_json(
                    self.upstream_base_url + _COMPLETIONS_SUFFIX, body, headers_out, self.request_timeout_s
                )
            except _ConnectionFailed as e:
                return _err(502, "upstream_unreachable", str(e))
            if status != 200:
                if not isinstance(data, dict):
                    data = {"error": {"message": str(data)[:2000], "code": "upstream_error"}}
                return status, data
            if not isinstance(data, dict):
                return _err(502, "bad_upstream_body", "upstream returned non-JSON")
            usage = parse_usage(data.get("usage"))
            if usage is None:
                return _err(502, "missing_usage", "upstream returned no usage field; cannot charge budget")
            run.calls.append({"role": role, "prompt_tokens": usage.prompt_tokens,
                              "completion_tokens": usage.completion_tokens, "cached_tokens": usage.cached_tokens,
                              "reasoning_tokens": usage.reasoning_tokens})
            try:
                budget.charge(role, usage.prompt_tokens, usage.completion_tokens,
                              cached_tokens=usage.cached_tokens, reasoning_tokens=usage.reasoning_tokens)
            except BudgetExhausted as e:
                return _err(402, BUDGET_EXHAUSTED, str(e), budget=budget.snapshot(), discarded=True)
            return 200, data

    def handle_models(self) -> tuple[int, dict]:
        model = self.model or "unknown"
        return 200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "proxy"}]}

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # keep test output quiet
                pass

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                if not self.path.rstrip("/").endswith(_COMPLETIONS_SUFFIX):
                    self._send(*_err(404, "not_found", f"unsupported path {self.path}"))
                    return
                try:
                    self._send(*proxy.handle_completion(self.headers, raw))
                except Exception as e:  # never kill the server thread
                    self._send(*_err(500, "proxy_error", f"{type(e).__name__}: {e}"))

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._send(*proxy.handle_models())
                else:
                    self._send(*_err(404, "not_found", f"unsupported path {self.path}"))

        return Handler


def _err(status: int, code: str, message: str, **extra: Any) -> tuple[int, dict]:
    return status, {"error": {"message": message, "type": code, "code": code, **extra}}


# ---------------------------------------------------------------------- client half


class ProxyLM:
    """`(str | list[dict]) -> str` callable that talks to an LLMProxy for one run.

    This matches gepa's `LanguageModel` protocol. It exposes `total_cost`,
    `total_tokens_in` and `total_tokens_out` (from the response `usage`, never estimated),
    so gepa uses it as-is instead of wrapping it in its char-count `TrackingLM`.

    A 402 `budget_exhausted` refusal raises `BudgetExhausted` and sets `self.exhausted`
    (gepa swallows exceptions raised during reflection, so callers should also poll the flag).
    A billing / quota refusal raises `BillingError` (never retried) and sets `self.billing_error`.
    429 / 5xx / connection errors are retried with exponential backoff.
    """

    def __init__(
        self,
        base_url: str,
        run_id: str,
        *,
        model: str = "proxy-default",
        role: str = "agent",
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 6,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        request_timeout_s: float = 900.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        on_response: Callable[[list[dict], str, dict], None] | None = None,
    ):
        self.url = base_url.rstrip("/") + _COMPLETIONS_SUFFIX
        self.run_id, self.model, self.role = run_id, model, role
        self.max_tokens, self.temperature = max_tokens, temperature
        self.max_retries, self.backoff_base_s, self.backoff_max_s = max_retries, backoff_base_s, backoff_max_s
        self._sleep = sleep_fn
        self.request_timeout_s = request_timeout_s
        self.consecutive_failures = 0  # non-budget errors in a row (reset on success)
        self.on_response = on_response
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.calls = 0
        self.exhausted = False
        self.billing_error: BillingError | None = None

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if self.billing_error is not None:
            raise self.billing_error
        if self.exhausted:
            raise BudgetExhausted("budget exhausted (proxy refused an earlier call)")
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {RUN_ID_HEADER: self.run_id, ROLE_HEADER: self.role}
        attempt = 0
        while True:
            try:
                status, payload = post_json(self.url, body, headers, self.request_timeout_s)
            except _ConnectionFailed as e:
                status, payload = -1, {"error": {"message": str(e), "code": "connection_failed"}}
            if status == 200 and isinstance(payload, dict):
                break
            err = (payload.get("error") or {}) if isinstance(payload, dict) else {}
            code = err.get("code") if isinstance(err, dict) else None
            if code == BUDGET_EXHAUSTED:
                self.exhausted = True
                raise BudgetExhausted(err.get("message", BUDGET_EXHAUSTED))
            billing = billing_error_from_body(status, payload)
            if billing is not None:  # quota / spend limit: retrying cannot help
                self.billing_error = billing
                raise billing
            retryable = status == -1 or status == 429 or status >= 500
            if retryable and code != "missing_usage" and attempt < self.max_retries:
                delay = min(self.backoff_max_s, self.backoff_base_s * 2**attempt)
                self._sleep(delay * (0.5 + random.random() / 2))
                attempt += 1
                continue
            self.consecutive_failures += 1
            raise ProxyError(status, code or "error", json.dumps(payload)[:2000])
        data = payload
        usage = data["usage"]
        self.consecutive_failures = 0
        self.calls += 1
        self.total_tokens_in += int(usage["prompt_tokens"])
        self.total_tokens_out += int(usage["completion_tokens"])
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if self.on_response is not None:
            self.on_response(messages, text, usage)
        return text

    def close(self) -> None:  # kept for API symmetry; urllib holds no connection pool
        pass


def main() -> None:  # pragma: no cover - manual debugging entry point
    import argparse

    from harness.config import load_dotenv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run-id", default="debug")
    ap.add_argument("--budget", type=int, default=100_000)
    args = ap.parse_args()
    load_dotenv()
    proxy = LLMProxy.from_env(port=args.port)
    proxy.register(args.run_id, Budget(args.budget))
    print(f"proxy at {proxy.base_url} (run_id {args.run_id!r}, budget {args.budget}); Ctrl-C to stop")
    proxy.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        proxy.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
