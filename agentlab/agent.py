"""The reference agent: a plain tool-use loop with step and cost limits."""
from __future__ import annotations

import time
from typing import Any

from .llm import LLMClient, cost_usd
from .models import Step, Trace
from .sandbox import Workspace

SYSTEM_PROMPT = (
    "You are a software engineering agent working inside a project directory. "
    "Use the tools to inspect and edit files and run commands. "
    "When the task is complete, reply with a short summary and no tool calls."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List all files in the workspace.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command in the workspace (60s timeout).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def execute_tool(ws: Workspace, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run one tool call. Returns (output, is_error). Never raises."""
    try:
        if name == "list_files":
            return "\n".join(ws.list()) or "(empty)", False
        if name == "read_file":
            return ws.read(args["path"]), False
        if name == "write_file":
            ws.write(args["path"], args["content"])
            return f"wrote {len(args['content'])} chars to {args['path']}", False
        if name == "run_shell":
            code, out = ws.run(args["command"])
            return f"[exit {code}]\n{out}", code != 0
        return f"unknown tool: {name}", True
    except Exception as e:  # tool errors go back to the model, not up the stack
        return f"{type(e).__name__}: {e}", True


def run_agent(
    llm: LLMClient,
    ws: Workspace,
    prompt: str,
    max_steps: int = 15,
    max_cost_usd: float = 1.0,
    system: str = SYSTEM_PROMPT,
) -> Trace:
    trace = Trace()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    spent = 0.0

    for i in range(max_steps):
        t0 = time.perf_counter()
        try:
            resp = llm.complete(system, messages, TOOLS)
        except Exception as e:
            trace.stop_reason, trace.error = "error", f"{type(e).__name__}: {e}"
            return trace
        latency = time.perf_counter() - t0
        spent += cost_usd(llm.model, resp.input_tokens, resp.output_tokens)

        step = Step(i, resp.text, [], resp.input_tokens, resp.output_tokens, latency)
        trace.steps.append(step)
        messages.append({"role": "assistant", "content": resp.to_blocks()})

        if not resp.tool_uses:
            trace.final_text, trace.stop_reason = resp.text, "done"
            return trace

        results = []
        for tu in resp.tool_uses:
            output, is_error = execute_tool(ws, tu.name, tu.input)
            step.tool_calls.append(
                {"name": tu.name, "input": tu.input, "output": output, "is_error": is_error}
            )
            results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": output, "is_error": is_error}
            )
        messages.append({"role": "user", "content": results})

        if spent >= max_cost_usd:
            trace.stop_reason = "cost_limit"
            return trace

    trace.stop_reason = "max_steps"
    return trace
