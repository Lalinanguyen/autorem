"""Small client for the Devin API (v3, organization scope) plus Claude tool definitions.

Docs: https://docs.devin.ai/api-reference/overview
Auth: `Authorization: Bearer $DEVIN_API_KEY`, where the key is a `cog_` service-user
key or personal access token with the UseDevinSessions permission.

Devin sessions run asynchronously and consume paid ACUs, so the tools here start a
session and return right away; Claude checks back with devin_get_session.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

BASE_URL = "https://api.devin.ai"
MESSAGE_CHARS = 4000  # truncate each Devin message in tool output
RECENT_MESSAGES = 10


class DevinError(Exception):
    """A Devin API call failed; the message is safe to show to the model (never contains the key)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DevinConfigError(DevinError):
    """DEVIN_API_KEY (or the organization id) is missing."""


class DevinAuthError(DevinError):
    """401/403: the key is invalid or lacks permission."""


class DevinNotFoundError(DevinError):
    """404: the session or organization does not exist."""


class DevinRateLimitError(DevinError):
    """429: too many requests."""

    def __init__(self, message: str, retry_after: str | None = None):
        super().__init__(message, 429)
        self.retry_after = retry_after


class DevinServerError(DevinError):
    """5xx: Devin had a server-side problem; retry later."""


def _problem_detail(response: httpx.Response) -> str:
    # v3 errors are RFC 7807 ProblemDetail bodies: {"title", "status", "detail", ...}.
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("title") or body)[:300]
    return str(body)[:300]


def _raise_for_status(response: httpx.Response) -> None:
    code = response.status_code
    if code < 400:
        return
    detail = _problem_detail(response)
    if code in (401, 403):
        raise DevinAuthError(
            f"Devin API returned {code}: {detail}. Check that DEVIN_API_KEY is a valid cog_ key "
            "with the UseDevinSessions permission for this organization.",
            code,
        )
    if code == 404:
        raise DevinNotFoundError(f"Devin API returned 404 (not found): {detail}", code)
    if code == 429:
        retry_after = response.headers.get("Retry-After")
        wait = f" Retry after {retry_after}s." if retry_after else ""
        raise DevinRateLimitError(f"Devin API rate limit hit (429): {detail}.{wait}", retry_after)
    if code >= 500:
        raise DevinServerError(f"Devin API server error {code}: {detail}. Try again later.", code)
    raise DevinError(f"Devin API returned {code}: {detail}", code)


class DevinClient:
    def __init__(
        self,
        api_key: str | None = None,
        org_id: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        api_key = api_key or os.environ.get("DEVIN_API_KEY")
        if not api_key:
            raise DevinConfigError("DEVIN_API_KEY is not set")
        self._org_id = org_id or os.environ.get("DEVIN_ORG_ID") or None
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            # httpx errors carry the URL, never the request headers.
            raise DevinError(f"Could not reach the Devin API: {error}") from error
        _raise_for_status(response)
        try:
            return response.json()
        except ValueError as error:
            raise DevinError(f"Devin API returned a non-JSON response ({response.status_code})") from error

    @property
    def org_id(self) -> str:
        """DEVIN_ORG_ID if set, else the organization the key belongs to (GET /v3/self)."""
        if not self._org_id:
            me = self._request("GET", "/v3/self")
            self._org_id = me.get("org_id") or me.get("devin_sessions_org_id")
            if not self._org_id:
                raise DevinConfigError(
                    "Could not determine the Devin organization for this key; set DEVIN_ORG_ID "
                    "(shown on Settings > Devin API)."
                )
        return self._org_id

    def _sessions(self, suffix: str = "") -> str:
        return f"/v3/organizations/{self.org_id}/sessions{suffix}"

    def create_session(self, prompt: str, *, title: str | None = None, max_acu_limit: int | None = None) -> dict[str, Any]:
        """Start a session. Returns the SessionResponse (session_id, url, status, ...)."""
        body: dict[str, Any] = {"prompt": prompt}
        if title:
            body["title"] = title
        if max_acu_limit is not None:
            body["max_acu_limit"] = max_acu_limit
        return self._request("POST", self._sessions(), json=body)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", self._sessions(f"/{session_id}"))

    def list_messages(self, session_id: str, first: int = 100) -> list[dict[str, Any]]:
        """Return the session's messages in chronological order (first page only, up to `first`)."""
        page = self._request("GET", self._sessions(f"/{session_id}/messages"), params={"first": first})
        return page.get("items", [])

    def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        """Send a follow-up message; Devin resumes the session if it was suspended."""
        return self._request("POST", self._sessions(f"/{session_id}/messages"), json={"message": message})


# ---- Claude tools ---------------------------------------------------------------

_COST_NOTE = (
    "Devin is Cognition's autonomous AI software engineer. It runs in its own cloud VM, "
    "asynchronously (minutes to hours), and every session consumes paid ACUs."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "devin_create_session",
        "description": (
            f"{_COST_NOTE} Start a new Devin session with a task prompt. Only use this when the user "
            "explicitly asks to delegate work to Devin, or approves delegating a large, self-contained "
            "task. Write a complete, well-scoped prompt: the repository, the goal, constraints, and what "
            "'done' looks like (e.g. open a PR). Returns immediately with the session id, URL and status; "
            "it does not wait for Devin to finish. Give the user the URL and check progress later with "
            "devin_get_session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The full task description for Devin."},
                "title": {"type": "string", "description": "Optional short session title."},
                "max_acu_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cap on ACUs this session may consume.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "devin_get_session",
        "description": (
            "Get the status of a Devin session: status (new, claimed, running, exit, error, suspended, "
            "resuming), status_detail (e.g. working, waiting_for_user, finished), ACUs consumed, pull "
            "requests, structured output, and Devin's most recent messages. Use it to check on a session "
            "you started; it does not wait. If Devin is still working, tell the user and check again later "
            "rather than calling this repeatedly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "The Devin session id (devin-...)."}},
            "required": ["session_id"],
        },
    },
    {
        "name": "devin_send_message",
        "description": (
            "Send a follow-up message to an existing Devin session, e.g. to answer a question when "
            "status_detail is waiting_for_user, or to add instructions. A suspended session resumes, "
            "which consumes more ACUs. Returns the updated session status without waiting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "The Devin session id (devin-...)."},
                "message": {"type": "string", "description": "The message to send to Devin."},
            },
            "required": ["session_id", "message"],
        },
    },
]

TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)

_SESSION_FIELDS = (
    "session_id", "url", "status", "status_detail", "title", "acus_consumed",
    "pull_requests", "structured_output", "created_at", "updated_at",
)


def _summary(session: dict[str, Any]) -> dict[str, Any]:
    return {k: session[k] for k in _SESSION_FIELDS if session.get(k) not in (None, [], {})}


def run_tool(client: DevinClient, name: str, args: dict[str, Any]) -> str:
    """Run one Devin tool call and return its result as JSON text. Raises DevinError on API failures."""
    if name == "devin_create_session":
        prompt = str(args["prompt"]).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        max_acu = args.get("max_acu_limit")
        if max_acu is None and os.environ.get("DEVIN_MAX_ACU_LIMIT"):
            max_acu = int(os.environ["DEVIN_MAX_ACU_LIMIT"])
        session = client.create_session(
            prompt, title=args.get("title"), max_acu_limit=int(max_acu) if max_acu is not None else None
        )
        result = _summary(session)
        result["note"] = "Devin is working asynchronously. Share the URL; check back with devin_get_session."
        return json.dumps(result, indent=2)
    if name == "devin_get_session":
        session_id = str(args["session_id"])
        result = _summary(client.get_session(session_id))
        messages = client.list_messages(session_id)[-RECENT_MESSAGES:]
        result["recent_messages"] = [
            {"source": m.get("source"), "message": str(m.get("message", ""))[:MESSAGE_CHARS]} for m in messages
        ]
        return json.dumps(result, indent=2)
    if name == "devin_send_message":
        message = str(args["message"]).strip()
        if not message:
            raise ValueError("message must not be empty")
        return json.dumps(_summary(client.send_message(str(args["session_id"]), message)), indent=2)
    raise ValueError(f"Unknown Devin tool {name!r}")
