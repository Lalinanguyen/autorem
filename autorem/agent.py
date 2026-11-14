"""A coding agent on Claude that can edit files in a workspace and search the web with Exa.

Usage:
    python -m autorem.agent "add a /health endpoint using the latest FastAPI docs"
    python -m autorem.agent --workspace path/to/project   # interactive

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile) and EXA_API_KEY.
Optional: DEVIN_API_KEY enables the Devin delegation tools (see autorem/devin.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import anthropic
from exa_py import Exa

from autorem import devin
from autorem.editor import TOOL as EDITOR_TOOL
from autorem.editor import EditorError, TextEditor

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are a coding agent working inside a single project workspace. You read and \
edit files with the str_replace_based_edit_tool, keeping changes small and \
consistent with the existing code.

You can search the live web with web_search and read specific pages with \
get_contents. Your training data may be out of date, so search before you rely \
on memory for:
- library or framework APIs, versions, and configuration options
- error messages you don't recognise
- anything released or changed recently

Every web_search call needs a `query` and an `objective`. Write the query as a \
natural-language description of the page you want, not just keywords. In the \
objective, say which sources should rank first (official docs, the project's \
own repo or changelog), what to exclude (outdated versions, SEO tutorials), and \
which facts or code you need from the results. When a result points to the \
exact docs page or file you need, call get_contents on its URL to read the full \
page before writing code against it.

When your answer depends on something you found on the web, mention the URL.

If devin_* tools are available, use them only when the user asks to delegate work \
to Devin (or approves it for a large, self-contained task): Devin sessions run \
asynchronously and cost money, so start one, share its URL, and check back later \
instead of polling."""


def build_tools(exa: Exa, devin_client: devin.DevinClient | None = None) -> tuple[list[Any], list[Any]]:
    """Return (all tool definitions, the Exa tools that exa.anthropic should run).

    The Devin tools are included only when a DevinClient is given (DEVIN_API_KEY is set).
    """
    exa_tools = [exa.anthropic.web_search(), exa.anthropic.get_contents()]
    for tool in exa_tools:
        # Stream tool inputs as they're generated. Inputs then arrive unvalidated,
        # but the Exa helpers validate them with pydantic before calling Exa.
        tool["eager_input_streaming"] = True
    devin_tools = devin.TOOLS if devin_client is not None else []
    return [*exa_tools, EDITOR_TOOL, *devin_tools], exa_tools


def run_tools(
    response: Any,
    exa: Exa,
    exa_tools: list[Any],
    editor: TextEditor,
    devin_client: devin.DevinClient | None = None,
) -> list[dict[str, Any]]:
    """Run every tool_use block in the response, returning all results in order."""
    exa_results = {r["tool_use_id"]: r for r in exa.anthropic.handle_tool_use(response, exa_tools)}
    results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        if block.name == EDITOR_TOOL["name"]:
            try:
                result = {"type": "tool_result", "tool_use_id": block.id, "content": editor.run(block.input)}
            except (EditorError, KeyError, ValueError, OSError) as error:
                result = {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {error}", "is_error": True}
            print(f"  [edit] {block.input.get('command')} {block.input.get('path')}", file=sys.stderr)
        elif block.name in devin.TOOL_NAMES:
            try:
                if devin_client is None:
                    raise devin.DevinConfigError("Devin tools are disabled: DEVIN_API_KEY is not set")
                result = {"type": "tool_result", "tool_use_id": block.id, "content": devin.run_tool(devin_client, block.name, block.input)}
            except (devin.DevinError, KeyError, ValueError, TypeError) as error:
                result = {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {error}", "is_error": True}
            print(f"  [{block.name}] {block.input.get('session_id') or str(block.input.get('prompt', ''))[:80]}", file=sys.stderr)
        else:
            result = exa_results[block.id]
            # The Exa helpers report failures as text starting with "Error:".
            if str(result["content"]).startswith("Error:"):
                result = {**result, "is_error": True}
            print(f"  [{block.name}] {block.input.get('query') or block.input.get('urls')}", file=sys.stderr)
        results.append(result)
    return results


def run_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
    tools: list[Any],
    exa: Exa,
    exa_tools: list[Any],
    editor: TextEditor,
    devin_client: devin.DevinClient | None = None,
) -> str:
    """Run the agent loop for the latest user message; return Claude's final text."""
    while True:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            cache_control={"type": "ephemeral"},
            # On a safety decline, the API reruns the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()

        # Append the full content (thinking, fallback and tool_use blocks), unchanged.
        messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            return text + f"\n[Claude declined this request (category: {category}).]"
        if response.stop_reason == "max_tokens":
            return text + "\n[Stopped: hit max_tokens before finishing.]"
        if response.stop_reason == "pause_turn":
            continue
        if response.stop_reason != "tool_use":
            return text

        if text:
            print(text, file=sys.stderr)
        messages.append({"role": "user", "content": run_tools(response, exa, exa_tools, editor, devin_client)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", help="Task to run. Omit for an interactive session.")
    parser.add_argument("--workspace", default=".", help="Directory the agent may read and edit (default: cwd).")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    client = anthropic.Anthropic()
    exa = Exa()  # reads EXA_API_KEY
    # Devin tools are optional: register them only when DEVIN_API_KEY is set.
    devin_client = devin.DevinClient() if os.environ.get("DEVIN_API_KEY") else None
    tools, exa_tools = build_tools(exa, devin_client)
    editor = TextEditor(workspace)
    messages: list[dict[str, Any]] = []

    def ask(prompt: str) -> None:
        messages.append({"role": "user", "content": f"Workspace root: {workspace}\n\n{prompt}" if not messages else prompt})
        print(run_turn(client, messages, tools, exa, exa_tools, editor, devin_client))

    if args.task:
        ask(args.task)
        return

    print(f"autorem coding agent - workspace {workspace}. Ctrl+C or empty line to quit.")
    try:
        while prompt := input("\n> ").strip():
            ask(prompt)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
