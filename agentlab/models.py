"""Core data models shared by the harness, agent and test generator."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    """One eval task: a starting workspace, a prompt, and a shell command that decides pass/fail."""

    id: str
    prompt: str
    check: str  # shell command run in the workspace after the agent finishes; exit 0 == pass
    files: dict[str, str] = field(default_factory=dict)  # relative path -> content
    max_steps: int = 15
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            check=d["check"],
            files=d.get("files", {}),
            max_steps=d.get("max_steps", 15),
            tags=d.get("tags", []),
        )


def load_tasks(path: str | Path) -> list[Task]:
    """Load tasks from a .json file (one task or a list) or a directory of .json files."""
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    tasks: list[Task] = []
    for f in files:
        data = json.loads(f.read_text())
        items = data if isinstance(data, list) else [data]
        tasks.extend(Task.from_dict(item) for item in items)
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate task ids: {sorted(dupes)}")
    return tasks


@dataclass
class Step:
    """One model turn plus the tool calls it made."""

    index: int
    text: str
    tool_calls: list[dict[str, Any]]  # {name, input, output, is_error}
    input_tokens: int
    output_tokens: int
    latency_s: float


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str = ""  # "done" | "max_steps" | "cost_limit" | "error"
    error: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)


@dataclass
class Result:
    """Outcome of one attempt at one task."""

    task_id: str
    attempt: int
    passed: bool
    check_output: str
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    stop_reason: str
    error: str | None = None
    trace: Trace | None = None

    def to_json(self, include_trace: bool = True) -> str:
        d = asdict(self)
        if not include_trace:
            d.pop("trace", None)
        return json.dumps(d)
