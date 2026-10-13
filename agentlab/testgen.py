"""Auto test generator: write pytest tests for a module, run them, feed failures back, repeat.

Failing tests are ambiguous: either the test is wrong or the code has a bug. The prompt tells the
model to treat the *current behavior as the spec only when it is consistent with the docstring/intent*,
and the report keeps the two cases visible instead of silently "fixing" tests to match buggy code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient
from .sandbox import Workspace

SYSTEM = (
    "You write pytest unit tests. Reply with exactly one ```python code block containing the full "
    "test file and nothing else. Import the module under test by its file name. "
    "Cover normal cases, edge cases (empty, zero, negative, None where plausible) and error paths. "
    "Use plain asserts and pytest.raises. No network, no randomness, no sleeping."
)

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    m = _CODE_RE.search(text)
    return m.group(1).strip() + "\n" if m else None


@dataclass
class Round:
    index: int
    passed: bool
    output: str
    coverage: float | None = None


@dataclass
class TestGenReport:
    source: str
    test_code: str
    rounds: list[Round] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].passed

    @property
    def coverage(self) -> float | None:
        return self.rounds[-1].coverage if self.rounds else None


_COV_RE = re.compile(r"TOTAL\s+\d+\s+\d+\s+(\d+)%")


def _run_tests(ws: Workspace, module: str) -> tuple[bool, str, float | None]:
    # Try with coverage; fall back to plain pytest when pytest-cov is not installed.
    code, out = ws.run(f"python -m pytest -q -x --no-header --cov={module} --cov-report=term test_{module}.py")
    if "unrecognized arguments: --cov" in out:
        code, out = ws.run(f"python -m pytest -q -x --no-header test_{module}.py")
    m = _COV_RE.search(out)
    return code == 0, out, (int(m.group(1)) / 100 if m else None)


def generate_tests(
    llm: LLMClient,
    source_path: str | Path,
    max_rounds: int = 3,
    target_coverage: float = 0.9,
    out_path: str | Path | None = None,
) -> TestGenReport:
    src_path = Path(source_path)
    module = src_path.stem
    source = src_path.read_text()
    report = TestGenReport(source=str(src_path), test_code="")

    prompt = f"Write pytest tests for this module, saved as `{src_path.name}`:\n\n```python\n{source}\n```"
    messages = [{"role": "user", "content": prompt}]

    with Workspace({src_path.name: source}) as ws:
        for i in range(max_rounds):
            resp = llm.complete(SYSTEM, messages, None)
            messages.append({"role": "assistant", "content": resp.text})
            code = extract_code(resp.text)
            if code is None:
                messages.append({"role": "user", "content": "No ```python block found. Reply with one code block."})
                report.rounds.append(Round(i, False, "model reply contained no code block"))
                continue

            report.test_code = code
            ws.write(f"test_{module}.py", code)
            ok, out, cov = _run_tests(ws, module)
            report.rounds.append(Round(i, ok, out, cov))

            if ok and (cov is None or cov >= target_coverage):
                break
            if ok:
                feedback = (
                    f"All tests pass but coverage is {cov:.0%} (target {target_coverage:.0%}). "
                    "Add tests for the uncovered lines and return the full updated file."
                )
            else:
                feedback = (
                    "Some tests failed:\n\n" + out + "\n\n"
                    "Decide for each failure whether the TEST is wrong or the CODE has a bug. "
                    "Fix wrong tests. If you believe the code is buggy, keep the test and add a "
                    "comment `# BUG:` explaining it. Return the full updated test file."
                )
            messages.append({"role": "user", "content": feedback})

    if out_path and report.test_code:
        Path(out_path).write_text(report.test_code)
    return report


def format_report(r: TestGenReport) -> str:
    lines = [f"source: {r.source}"]
    for rd in r.rounds:
        cov = f" coverage={rd.coverage:.0%}" if rd.coverage is not None else ""
        lines.append(f"round {rd.index + 1}: {'PASS' if rd.passed else 'FAIL'}{cov}")
    lines.append(f"final: {'PASS' if r.passed else 'FAIL'}" + (f" coverage={r.coverage:.0%}" if r.coverage is not None else ""))
    return "\n".join(lines)
