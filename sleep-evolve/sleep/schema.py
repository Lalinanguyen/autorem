"""Optional schema store (FullSleep flag `schema`, default off).

In REM sleeps only, one role="sleep" LLM call abstracts the strong memory items into a few
schemas: general patterns that several items instantiate. Schemas are rendered (after goals,
before items) inside the SAME memory token cap, so they never grow the agent context.

Persistence to disk is off by default: a schema file reused across runs leaks knowledge
between runs/seeds and breaks independence of the comparison. Enable only deliberately.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sleep.consolidator_io import parse_json_object
from sleep.memory import MemoryStore, normalize_tags

SCHEMA_SYSTEM = (
    "You abstract an agent's long-term memory into memory schemas: general, reusable patterns "
    "that several memory items are instances of. You never work on the problem directly."
)

SCHEMA_USER = """<task>{task}</task>

<memory_items>
{items}
</memory_items>

Write at most {max_schemas} schemas. Each must generalize at least two of the items above
and name them in "members". Output JSON only:
{{"schemas": [{{"text": "...", "approach_tags": [...], "members": [memory ids]}}]}}"""


@dataclass
class Schema:
    id: str
    text: str
    approach_tags: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    created_step: int = 0


class SchemaStore:
    def __init__(self, *, min_strength: float = 0.6, min_items: int = 3, max_schemas: int = 4,
                 max_tokens: int = 2048, max_chars: int = 300, path: str | None = None, persist: bool = False):
        self.min_strength = min_strength
        self.min_items = min_items
        self.max_schemas = max_schemas
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.path = Path(path) if (path and persist) else None
        self.schemas: list[Schema] = []
        self._next = 1
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.schemas = [Schema(**s) for s in data.get("schemas", [])]
            self._next = int(data.get("next", len(self.schemas) + 1))

    def lines(self) -> list[str]:
        return [f"- [{s.id}] SCHEMA: {s.text}" + (f" [tags: {', '.join(s.approach_tags)}]" if s.approach_tags else "")
                for s in self.schemas]

    def update(self, task: str, store: MemoryStore, llm, *, step: int) -> dict:
        """REM-only abstraction. Returns metrics. Malformed output keeps the old schemas."""
        strong = [i for i in store.ordered() if i.kind != "goal" and i.status != "refuted"
                  and i.strength >= self.min_strength]
        if len(strong) < self.min_items:
            return {"schema_ran": False, "schemas": len(self.schemas)}
        items = "\n".join(json.dumps({"id": i.id, "kind": i.kind, "text": i.text, "approach_tags": i.approach_tags})
                          for i in strong)
        user = SCHEMA_USER.format(task=task, items=items, max_schemas=self.max_schemas)
        resp = llm.complete([{"role": "system", "content": SCHEMA_SYSTEM}, {"role": "user", "content": user}],
                            role="sleep", max_tokens=self.max_tokens)
        obj, err = parse_json_object(resp.text)
        raw = obj.get("schemas") if isinstance(obj, dict) else None
        if not isinstance(raw, list):
            return {"schema_ran": True, "schema_parse_error": err or "no schemas list", "schemas": len(self.schemas)}
        valid_ids = {i.id for i in strong}
        new: list[Schema] = []
        for s in raw[: self.max_schemas]:
            if not isinstance(s, dict) or not isinstance(s.get("text"), str) or not s["text"].strip():
                continue
            members = [m for m in s.get("members", []) if isinstance(m, str) and m in valid_ids]
            if len(members) < 2:
                continue
            new.append(Schema(id=f"s{self._next}", text=" ".join(s["text"].split())[: self.max_chars],
                              approach_tags=normalize_tags(s.get("approach_tags")), members=members,
                              created_step=step))
            self._next += 1
        if new:
            self.schemas = new
            self.save()
        return {"schema_ran": True, "schemas": len(self.schemas), "schemas_added": len(new)}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"next": self._next, "schemas": [asdict(s) for s in self.schemas]}),
                             encoding="utf-8")
