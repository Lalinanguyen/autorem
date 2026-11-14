"""Offline smoke tests for the Devin client and its run_tools wiring (no network, no real key).

Run: python -m tests.test_devin   (or with pytest)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from autorem import devin
from autorem.agent import build_tools, run_tools
from autorem.editor import TextEditor

FAKE_KEY = "cog_test_not_a_real_key"
ORG = "org-abc123"
SESSION = {
    "session_id": "devin-123",
    "url": "https://app.devin.ai/sessions/123",
    "status": "running",
    "status_detail": "working",
    "tags": [],
    "org_id": ORG,
    "created_at": 1,
    "updated_at": 2,
    "acus_consumed": 0.5,
    "pull_requests": [],
}


def make_client(handler, org_id: str | None = ORG) -> tuple[devin.DevinClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = devin.DevinClient(FAKE_KEY, org_id, transport=httpx.MockTransport(record))
    return client, seen


def ok_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if request.method == "GET" and path == "/v3/self":
        return httpx.Response(200, json={"principal_type": "service_user", "service_user_id": "su", "service_user_name": "bot", "org_id": ORG})
    if request.method == "POST" and path == f"/v3/organizations/{ORG}/sessions":
        return httpx.Response(200, json=SESSION)
    if request.method == "GET" and path == f"/v3/organizations/{ORG}/sessions/devin-123":
        return httpx.Response(200, json={**SESSION, "status_detail": "finished"})
    if request.method == "GET" and path == f"/v3/organizations/{ORG}/sessions/devin-123/messages":
        items = [{"event_id": "e1", "source": "user", "message": "do it", "created_at": 1},
                 {"event_id": "e2", "source": "devin", "message": "done, PR opened", "created_at": 2}]
        return httpx.Response(200, json={"items": items, "end_cursor": None, "has_next_page": False})
    if request.method == "POST" and path == f"/v3/organizations/{ORG}/sessions/devin-123/messages":
        return httpx.Response(200, json=SESSION)
    return httpx.Response(404, json={"title": "Not Found", "status": 404, "detail": f"no route {path}"})


def test_create_get_send() -> None:
    client, seen = make_client(ok_handler)
    created = client.create_session("fix the bug", title="t", max_acu_limit=5)
    assert created["session_id"] == "devin-123" and created["url"].startswith("https://")
    req = seen[-1]
    assert req.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert str(req.url) == f"https://api.devin.ai/v3/organizations/{ORG}/sessions"
    assert json.loads(req.content) == {"prompt": "fix the bug", "title": "t", "max_acu_limit": 5}

    assert client.get_session("devin-123")["status_detail"] == "finished"
    msgs = client.list_messages("devin-123")
    assert seen[-1].url.params["first"] == "100" and msgs[-1]["source"] == "devin"

    client.send_message("devin-123", "add tests")
    assert json.loads(seen[-1].content) == {"message": "add tests"}


def test_org_id_from_self() -> None:
    saved = os.environ.pop("DEVIN_ORG_ID", None)
    try:
        client, seen = make_client(ok_handler, org_id=None)
        client.create_session("x")
        assert [r.url.path for r in seen] == ["/v3/self", f"/v3/organizations/{ORG}/sessions"]
    finally:
        if saved is not None:
            os.environ["DEVIN_ORG_ID"] = saved


def expect(exc_type: type[Exception], status: int, headers: dict[str, str] | None = None) -> devin.DevinError:
    body = {"title": "Problem", "status": status, "detail": "boom"}
    client, _ = make_client(lambda r: httpx.Response(status, json=body, headers=headers))
    try:
        client.get_session("devin-123")
    except exc_type as error:
        assert FAKE_KEY not in str(error), "API key leaked into error message"
        return error
    raise AssertionError(f"expected {exc_type.__name__} for {status}")


def test_errors() -> None:
    assert expect(devin.DevinAuthError, 401).status_code == 401
    expect(devin.DevinAuthError, 403)
    expect(devin.DevinNotFoundError, 404)
    assert expect(devin.DevinRateLimitError, 429, {"Retry-After": "30"}).retry_after == "30"
    assert expect(devin.DevinServerError, 503).status_code == 503
    assert "422" in str(expect(devin.DevinError, 422))

    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, _ = make_client(raise_connect)
    try:
        client.get_session("x")
        raise AssertionError("expected DevinError")
    except devin.DevinError as error:
        assert "Could not reach" in str(error)

    saved = os.environ.pop("DEVIN_API_KEY", None)
    try:
        devin.DevinClient()
        raise AssertionError("expected DevinConfigError")
    except devin.DevinConfigError:
        pass
    finally:
        if saved is not None:
            os.environ["DEVIN_API_KEY"] = saved


# ---- run_tools routing -----------------------------------------------------------


class FakeExaAnthropic:
    """Mimics exa.anthropic: answers every tool_use block, with an error for non-Exa tools."""

    def handle_tool_use(self, response: Any, tools: Any) -> list[dict[str, Any]]:
        out = []
        for b in response.content:
            if b.type != "tool_use":
                continue
            content = "search results" if b.name == "web_search" else f'Error: unknown tool "{b.name}"'
            out.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
        return out

    def web_search(self) -> dict[str, Any]:
        return {"name": "web_search", "input_schema": {"type": "object"}}

    def get_contents(self) -> dict[str, Any]:
        return {"name": "get_contents", "input_schema": {"type": "object"}}


def tool_use(id_: str, name: str, input_: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def test_build_tools() -> None:
    exa = SimpleNamespace(anthropic=FakeExaAnthropic())
    names = lambda tools: [t["name"] for t in tools]  # noqa: E731
    tools, _ = build_tools(exa)
    assert not devin.TOOL_NAMES & set(names(tools)), "Devin tools registered without a client"
    client, _ = make_client(ok_handler)
    tools, _ = build_tools(exa, client)
    assert devin.TOOL_NAMES <= set(names(tools))


def test_run_tools_routing(tmp_path: Path) -> None:
    exa = SimpleNamespace(anthropic=FakeExaAnthropic())
    editor = TextEditor(tmp_path)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    response = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="working"),
        tool_use("t1", "web_search", {"query": "q", "objective": "o"}),
        tool_use("t2", "str_replace_based_edit_tool", {"command": "view", "path": "a.txt"}),
        tool_use("t3", "devin_create_session", {"prompt": "refactor module X and open a PR"}),
        tool_use("t4", "devin_get_session", {"session_id": "devin-123"}),
        tool_use("t5", "devin_send_message", {"session_id": "devin-123", "message": "thanks"}),
        tool_use("t6", "devin_get_session", {}),  # missing session_id
        tool_use("t7", "devin_create_session", {"prompt": "   "}),  # empty prompt
    ])
    client, _ = make_client(ok_handler)
    results = run_tools(response, exa, [], editor, client)
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    assert results[0]["content"] == "search results" and not results[0].get("is_error")
    assert "hello" in results[1]["content"] and not results[1].get("is_error")
    created = json.loads(results[2]["content"])
    assert created["session_id"] == "devin-123" and created["url"] and not results[2].get("is_error")
    got = json.loads(results[3]["content"])
    assert got["status_detail"] == "finished" and got["recent_messages"][-1]["message"] == "done, PR opened"
    assert json.loads(results[4]["content"])["status"] == "running"
    assert results[5]["is_error"] and results[6]["is_error"]

    # API failure -> is_error tool result, loop keeps going.
    failing, _ = make_client(lambda r: httpx.Response(429, json={"title": "Too Many Requests", "status": 429}))
    [res] = run_tools(SimpleNamespace(content=[tool_use("t8", "devin_get_session", {"session_id": "x"})]), exa, [], editor, failing)
    assert res["is_error"] and "429" in res["content"]

    # Devin tool called with no client configured -> is_error, no crash.
    [res] = run_tools(SimpleNamespace(content=[tool_use("t9", "devin_create_session", {"prompt": "x"})]), exa, [], editor, None)
    assert res["is_error"] and "DEVIN_API_KEY" in res["content"]


if __name__ == "__main__":
    import tempfile

    test_create_get_send()
    test_org_id_from_self()
    test_errors()
    test_build_tools()
    with tempfile.TemporaryDirectory() as d:
        test_run_tools_routing(Path(d))
    print("all Devin smoke tests passed")
