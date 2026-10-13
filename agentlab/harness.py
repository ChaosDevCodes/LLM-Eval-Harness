"""Run tasks, record results, summarize, and diff runs."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .agent import run_agent
from .llm import LLMClient, cost_usd
from .models import Result, Task
from .sandbox import Workspace


def run_attempt(llm: LLMClient, task: Task, attempt: int, max_cost_usd: float = 1.0) -> Result:
    t0 = time.perf_counter()
    with Workspace(task.files) as ws:
        trace = run_agent(llm, ws, task.prompt, max_steps=task.max_steps, max_cost_usd=max_cost_usd)
        code, out = ws.run(task.check, timeout=120)
    return Result(
        task_id=task.id,
        attempt=attempt,
        passed=(code == 0 and trace.stop_reason != "error"),
        check_output=out,
        steps=len(trace.steps),
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        cost_usd=cost_usd(llm.model, trace.input_tokens, trace.output_tokens),
        latency_s=time.perf_counter() - t0,
        stop_reason=trace.stop_reason,
        error=trace.error,
        trace=trace,
    )


def run_eval(
    llm: LLMClient,
    tasks: list[Task],
    attempts: int = 1,
    workers: int = 1,
    out_path: str | Path | None = None,
    max_cost_usd: float = 1.0,
) -> list[Result]:
    jobs = [(t, a) for t in tasks for a in range(attempts)]
    out = Path(out_path) if out_path else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")

    results: list[Result] = []

    def work(job):
        task, attempt = job
        return run_attempt(llm, task, attempt, max_cost_usd)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for r in pool.map(work, jobs):
            results.append(r)
            if out:
                with out.open("a") as f:
                    f.write(r.to_json() + "\n")
    return results


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@dataclass
class Summary:
    tasks: int
    attempts: int
    pass_rate: float
    pass_at_k: float  # fraction of tasks solved by at least one attempt
    mean_steps: float
    total_cost_usd: float
    mean_latency_s: float
    p95_latency_s: float


def summarize(results: list[dict]) -> Summary:
    if not results:
        return Summary(0, 0, 0, 0, 0, 0, 0, 0)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)
    lat = sorted(r["latency_s"] for r in results)
    n = len(results)
    return Summary(
        tasks=len(by_task),
        attempts=n,
        pass_rate=sum(r["passed"] for r in results) / n,
        pass_at_k=sum(any(x["passed"] for x in rs) for rs in by_task.values()) / len(by_task),
        mean_steps=sum(r["steps"] for r in results) / n,
        total_cost_usd=sum(r["cost_usd"] for r in results),
        mean_latency_s=sum(lat) / n,
        p95_latency_s=lat[min(n - 1, int(0.95 * n))],
    )


def compare(a: list[dict], b: list[dict]) -> dict:
    """Per-task diff between two runs. A task 'passes' in a run if its majority of attempts pass."""

    def task_pass(rs: list[dict]) -> dict[str, bool]:
        g: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            g[r["task_id"]].append(r["passed"])
        return {t: sum(v) * 2 > len(v) for t, v in g.items()}

    pa, pb = task_pass(a), task_pass(b)
    common = sorted(set(pa) & set(pb))
    return {
        "regressions": [t for t in common if pa[t] and not pb[t]],
        "improvements": [t for t in common if not pa[t] and pb[t]],
        "unchanged_pass": [t for t in common if pa[t] and pb[t]],
        "unchanged_fail": [t for t in common if not pa[t] and not pb[t]],
        "only_in_a": sorted(set(pa) - set(pb)),
        "only_in_b": sorted(set(pb) - set(pa)),
        "summary_a": summarize(a),
        "summary_b": summarize(b),
    }


def format_summary(s: Summary) -> str:
    return (
        f"tasks={s.tasks} attempts={s.attempts} pass_rate={s.pass_rate:.0%} pass@k={s.pass_at_k:.0%} "
        f"steps={s.mean_steps:.1f} cost=${s.total_cost_usd:.4f} "
        f"latency mean={s.mean_latency_s:.2f}s p95={s.p95_latency_s:.2f}s"
    )
