"""Client-side handler for Claude's text editor tool, confined to one workspace root."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


class EditorError(Exception):
    """Raised for bad tool input; the message is shown to the model."""


class TextEditor:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _resolve(self, raw: str) -> Path:
        # `path` is untrusted model output: resolve it and require it to stay under root.
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise EditorError(f"Path {raw!r} is outside the workspace {self.root}")
        return resolved

    def run(self, args: dict[str, Any]) -> str:
        command = args.get("command")
        path = self._resolve(str(args.get("path", "")))
        if command == "view":
            return self._view(path, args.get("view_range"))
        if command == "create":
            return self._create(path, args["file_text"])
        if command == "str_replace":
            return self._str_replace(path, args["old_str"], args.get("new_str", ""))
        if command == "insert":
            return self._insert(path, int(args["insert_line"]), args["insert_text"])
        raise EditorError(f"Unknown command {command!r}")

    def _view(self, path: Path, view_range: list[int] | None) -> str:
        if path.is_dir():
            entries = sorted(
                p.relative_to(self.root).as_posix() + ("/" if p.is_dir() else "")
                for p in path.iterdir()
                if p.name != ".git"
            )
            return "\n".join(entries) or "(empty directory)"
        if not path.is_file():
            raise EditorError(f"{path} does not exist")
        lines = path.read_text(encoding="utf-8").splitlines()
        start, end = 1, len(lines)
        if view_range:
            start = max(1, view_range[0])
            end = len(lines) if view_range[1] == -1 else min(len(lines), view_range[1])
        return "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))

    def _create(self, path: Path, text: str) -> str:
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"Wrote {path.relative_to(self.root).as_posix()}"

    def _str_replace(self, path: Path, old: str, new: str) -> str:
        if not path.is_file():
            raise EditorError(f"{path} does not exist")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise EditorError(f"old_str must match exactly once; found {count} matches")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return "Replaced 1 occurrence"

    def _insert(self, path: Path, line: int, text: str) -> str:
        if not path.is_file():
            raise EditorError(f"{path} does not exist")
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not 0 <= line <= len(lines):
            raise EditorError(f"insert_line must be between 0 and {len(lines)}")
        if not text.endswith("\n"):
            text += "\n"
        lines.insert(line, text)
        path.write_text("".join(lines), encoding="utf-8")
        return f"Inserted after line {line}"
