"""Tests for agentlab itself, using the scripted MockLLM so no API key or network is needed."""
import json

from agentlab.agent import run_agent
from agentlab.harness import compare, run_eval, summarize
from agentlab.llm import LLMResponse, MockLLM, ToolUse
from agentlab.models import Task, load_tasks
from agentlab.sandbox import Workspace
from agentlab.testgen import extract_code, generate_tests


def tool(name, **inp):
    return LLMResponse(tool_uses=[ToolUse(id=f"t-{name}", name=name, input=inp)], input_tokens=10, output_tokens=5)


def done(text="done"):
    return LLMResponse(text=text, input_tokens=10, output_tokens=5)


# ---- sandbox -------------------------------------------------------------

def test_workspace_blocks_path_escape():
    with Workspace() as ws:
        try:
            ws.write("../evil.txt", "x")
        except ValueError:
            return
        raise AssertionError("path escape was allowed")


def test_workspace_timeout_kills_command():
    with Workspace() as ws:
        code, out = ws.run("sleep 5", timeout=0.5)
        assert code == 124 and "timed out" in out


def test_workspace_strips_secrets_from_env(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "shh")
    with Workspace() as ws:
        _, out = ws.run("echo [$FAKE_API_KEY]")
        assert "shh" not in out


# ---- agent ---------------------------------------------------------------

def test_agent_writes_file_then_finishes():
    llm = MockLLM([tool("write_file", path="a.txt", content="hi"), done()])
    with Workspace() as ws:
        trace = run_agent(llm, ws, "make a.txt")
        assert ws.read("a.txt") == "hi"
    assert trace.stop_reason == "done" and len(trace.steps) == 2


def test_agent_stops_at_max_steps():
    llm = MockLLM([tool("list_files")])  # repeats forever
    with Workspace() as ws:
        trace = run_agent(llm, ws, "loop", max_steps=3)
    assert trace.stop_reason == "max_steps" and len(trace.steps) == 3


def test_agent_tool_error_goes_back_to_model_not_raised():
    llm = MockLLM([tool("read_file", path="missing.txt"), done()])
    with Workspace() as ws:
        trace = run_agent(llm, ws, "x")
    assert trace.steps[0].tool_calls[0]["is_error"] is True
    assert trace.stop_reason == "done"


# ---- harness -------------------------------------------------------------

TASK = Task(id="t1", prompt="make ok.txt", check="test -f ok.txt")


def test_run_eval_pass_and_fail(tmp_path):
    good = MockLLM([tool("write_file", path="ok.txt", content="1"), done()])
    bad = MockLLM([done()])
    out_good, out_bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
    rg = run_eval(good, [TASK], out_path=out_good)
    rb = run_eval(bad, [TASK], out_path=out_bad)
    assert rg[0].passed and not rb[0].passed

    lines = [json.loads(line) for line in out_good.read_text().splitlines()]
    assert lines[0]["task_id"] == "t1" and lines[0]["trace"]["steps"]

    d = compare(
        [json.loads(line) for line in out_good.read_text().splitlines()],
        [json.loads(line) for line in out_bad.read_text().splitlines()],
    )
    assert d["regressions"] == ["t1"]


def test_summarize_pass_at_k():
    rows = [
        {"task_id": "a", "passed": False, "steps": 1, "cost_usd": 0.1, "latency_s": 1.0},
        {"task_id": "a", "passed": True, "steps": 1, "cost_usd": 0.1, "latency_s": 2.0},
        {"task_id": "b", "passed": False, "steps": 1, "cost_usd": 0.1, "latency_s": 3.0},
    ]
    s = summarize(rows)
    assert s.pass_at_k == 0.5 and abs(s.pass_rate - 1 / 3) < 1e-9


def test_sample_tasks_load():
    from pathlib import Path

    tasks = load_tasks(Path(__file__).parent.parent / "tasks")
    assert {t.id for t in tasks} >= {"fix-off-by-one", "add-function", "rename-and-update"}


# ---- testgen -------------------------------------------------------------

def test_extract_code():
    assert extract_code("hi\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert extract_code("no code here") is None


def test_testgen_recovers_from_failing_first_round(tmp_path):
    src = tmp_path / "calc.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    wrong = "```python\nfrom calc import add\n\ndef test_add():\n    assert add(1, 2) == 4\n```"
    right = "```python\nfrom calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n```"
    llm = MockLLM([LLMResponse(text=wrong), LLMResponse(text=right)])
    out = tmp_path / "test_calc.py"
    rep = generate_tests(llm, src, max_rounds=3, out_path=out)
    assert [r.passed for r in rep.rounds] == [False, True]
    assert rep.passed and out.exists()
