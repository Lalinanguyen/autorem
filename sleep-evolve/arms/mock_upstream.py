"""A fake OpenAI-compatible upstream for offline tests and `--mock-llm` runs. No network.

`FakeUpstream(responder)` serves `POST /v1/chat/completions` on 127.0.0.1 and returns
`responder(messages, body) -> (text, prompt_tokens, completion_tokens)` with a matching
`usage` field. Every request body is recorded in `.requests`.

Responders:
  - `p36_responder(seed)`: valid p36 programs (jittered 6x5 grid, shrunken radii), with an
    occasional overlapping one so side_info has something to report; notes text for
    non-proposal (e.g. sleep) prompts.
  - `mock_transport_responder(MockTransport(...))`: the existing deterministic mock
    (problem 0 programs) behind HTTP.
"""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

Responder = Callable[[list[dict], dict], tuple[str, int, int]]

# Markers of a GEPA reflection (proposal) prompt; see gepa/optimize_anything.py
# (_build_reflection_prompt_template and optimize_anything_reflection_prompt_template).
PROPOSAL_MARKERS = ("## Current Component", "I am optimizing a parameter")


def is_proposal_request(messages: list[dict]) -> bool:
    text = "\n".join(str(m.get("content", "")) for m in messages)
    return any(m in text for m in PROPOSAL_MARKERS)


def p36_program(rng: random.Random, overlap: bool = False) -> str:
    cols, rows, n = 6, 5, 26
    centers, radii = [], []
    for k in range(n):
        i, j = divmod(k, cols)
        centers.append([round((j + 0.5) / cols, 6), round((i + 0.5) / rows, 6)])
        radii.append(round((1 / 12) * rng.uniform(0.85, 0.999), 6))  # <= 1/12: always valid
    if overlap:
        radii[rng.randrange(n)] = 0.1
    return (
        "Shrink-and-jitter variant.\n```python\nimport numpy as np\n\n\ndef construct():\n"
        f"    centers = np.array({centers})\n    radii = np.array({radii})\n"
        "    return centers, radii\n```"
    )


def p36_responder(seed: int = 0, prompt_tokens: int = 60, completion_tokens: int = 40,
                  overlap_every: int = 4) -> Responder:
    rng = random.Random(seed)
    count = {"n": 0}

    def respond(messages: list[dict], body: dict) -> tuple[str, int, int]:
        count["n"] += 1
        if is_proposal_request(messages):
            text = p36_program(rng, overlap=overlap_every > 0 and count["n"] % overlap_every == 0)
        else:
            text = f"Consolidated notes after {count['n']} calls: keep radii <= 1/12 on the 6x5 grid."
        return text, prompt_tokens, completion_tokens

    return respond


def mock_transport_responder(transport: Any) -> Responder:
    """Adapt llm.mock.MockTransport (or any Transport) to a Responder."""

    def respond(messages: list[dict], body: dict) -> tuple[str, int, int]:
        r = transport(messages, int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096))
        return r.text, r.prompt_tokens, r.completion_tokens

    return respond


class FakeUpstream:
    def __init__(self, responder: Responder, *, omit_usage: bool = False, status: int = 200):
        self.responder = responder
        self.omit_usage = omit_usage
        self.status = status  # set to e.g. 500 to simulate an outage
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "FakeUpstream":
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True, name="fake-upstream"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()

    def __enter__(self) -> "FakeUpstream":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with fake._lock:
                    fake.requests.append(body)
                    fake.headers.append(dict(self.headers))
                    if fake.status != 200:
                        payload, status = {"error": {"message": "simulated outage", "code": "server_error"}}, fake.status
                    else:
                        text, pt, ct = fake.responder(body.get("messages", []), body)
                        payload, status = {
                            "id": f"chatcmpl-fake-{len(fake.requests)}",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body.get("model", "fake"),
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": text}}],
                        }, 200
                        if not fake.omit_usage:
                            payload["usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler
