"""Command-line interface: agentlab run | compare | testgen."""
from __future__ import annotations

import argparse
import json
import sys

from .harness import compare, format_summary, load_results, run_eval, summarize
from .llm import DEFAULT_MODEL, make_client
from .models import load_tasks
from .testgen import format_report, generate_tests


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an eval over a task file or directory")
    r.add_argument("tasks")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--attempts", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--max-cost", type=float, default=1.0, help="per-attempt USD cap")
    r.add_argument("--out", default="runs/latest.jsonl")

    c = sub.add_parser("compare", help="diff two result files")
    c.add_argument("a")
    c.add_argument("b")

    s = sub.add_parser("summary", help="summarize one result file")
    s.add_argument("results")

    t = sub.add_parser("testgen", help="generate pytest tests for a Python file")
    t.add_argument("source")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--target-coverage", type=float, default=0.9)
    t.add_argument("--out", help="where to save the final test file")

    args = p.parse_args(argv)

    if args.cmd == "run":
        tasks = load_tasks(args.tasks)
        results = run_eval(
            make_client(args.model), tasks, args.attempts, args.workers, args.out, args.max_cost
        )
        for res in results:
            print(f"{'PASS' if res.passed else 'FAIL'}  {res.task_id}#{res.attempt}  steps={res.steps} ${res.cost_usd:.4f}")
        print(format_summary(summarize([json.loads(res.to_json(False)) for res in results])))
        print(f"traces: {args.out}")
        return 0

    if args.cmd == "summary":
        print(format_summary(summarize(load_results(args.results))))
        return 0

    if args.cmd == "compare":
        d = compare(load_results(args.a), load_results(args.b))
        print("A:", format_summary(d["summary_a"]))
        print("B:", format_summary(d["summary_b"]))
        for key in ("regressions", "improvements", "unchanged_fail", "only_in_a", "only_in_b"):
            if d[key]:
                print(f"{key}: {', '.join(d[key])}")
        return 1 if d["regressions"] else 0

    if args.cmd == "testgen":
        rep = generate_tests(make_client(args.model), args.source, args.rounds, args.target_coverage, args.out)
        print(format_report(rep))
        return 0 if rep.passed else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
