# agentlab

A small eval harness for LLM coding agents, plus an auto test generator.

- **Harness**: define tasks (starting files + prompt + a shell command that decides pass/fail), run an agent against them in throwaway workspaces, record full traces, tokens, cost and latency, and diff two runs to catch regressions.
- **Test generator**: given a Python file, ask a model for pytest tests, run them, feed failures back, and repeat until they pass with enough coverage.

## Install

```bash
pip install -e '.[dev,anthropic]'
export ANTHROPIC_API_KEY=...
```

## Use

```bash
# run the sample tasks, 3 attempts each, 4 in parallel
agentlab run tasks/ --attempts 3 --workers 4 --out runs/baseline.jsonl

# change something (model, prompt, tools), run again, then diff
agentlab run tasks/ --attempts 3 --out runs/candidate.jsonl
agentlab compare runs/baseline.jsonl runs/candidate.jsonl   # exit 1 if any regression

# try it offline with the built-in mock model (no API key, solves 2 of the 3 sample tasks)
agentlab run tasks/ --model mock --out runs/mock.jsonl

# generate tests
agentlab testgen path/to/module.py --rounds 3 --out test_module.py
```

## Task format

```json
{
  "id": "fix-off-by-one",
  "prompt": "Fix last_n in utils.py so pytest passes.",
  "files": {"utils.py": "...", "test_utils.py": "..."},
  "check": "python -m pytest -q",
  "max_steps": 15
}
```

`check` runs in the workspace after the agent stops; exit code 0 means pass. Prefer checks the agent cannot satisfy by editing the tests (keep tests out of `files` and add them in `check`, or grep for tampering).

## Layout

| File | Role |
| --- | --- |
| `agentlab/llm.py` | `LLMClient` interface, `MockLLM` for offline tests, `AnthropicLLM`, pricing table |
| `agentlab/sandbox.py` | temp workspace, path-escape guard, timeouts, secret-stripped env |
| `agentlab/agent.py` | tool-use loop with step and cost limits |
| `agentlab/harness.py` | run tasks, JSONL results, summary (pass rate, pass@k, cost, p95), `compare` |
| `agentlab/testgen.py` | generate, run, feed back, coverage |
| `tests/` | tests for the harness itself using the mock model |

## Known limits and next steps

- The sandbox is not a security boundary. Run the whole harness inside Docker before pointing it at untrusted tasks.
- `PRICING` in `llm.py` holds placeholder numbers; edit it to your current rates.
- Ideas: bring in a slice of SWE-bench Lite as tasks, add an HTML report, add mutation testing to score generated tests (do they catch injected bugs?), cache model calls for cheaper reruns.
