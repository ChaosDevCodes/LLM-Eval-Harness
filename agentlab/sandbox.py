"""Workspace + command runner.

NOTE: this isolates the working directory, enforces timeouts and kills process groups,
but it is NOT a security boundary. For untrusted code, run the whole harness in Docker.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

MAX_OUTPUT_CHARS = 8000


def _truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(s) <= limit:
        return s
    half = limit // 2
    return s[:half] + f"\n... [{len(s) - limit} chars truncated] ...\n" + s[-half:]


class Workspace:
    def __init__(self, files: dict[str, str] | None = None):
        self.root = Path(tempfile.mkdtemp(prefix="agentlab-")).resolve()
        for rel, content in (files or {}).items():
            self.write(rel, content)

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes workspace: {rel}")
        return p

    def read(self, rel: str) -> str:
        return self.resolve(rel).read_text()

    def write(self, rel: str, content: str) -> None:
        p = self.resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list(self) -> list[str]:
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                out.append(str(p.relative_to(self.root)))
        return out

    def run(self, cmd: str, timeout: float = 60.0) -> tuple[int, str]:
        """Run a shell command in the workspace. Returns (exit_code, combined_output). Exit 124 on timeout."""
        env = {k: v for k, v in os.environ.items() if not k.endswith(("_KEY", "_TOKEN", "_SECRET"))}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            return proc.returncode, _truncate(out or "")
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return 124, _truncate((out or "") + f"\n[timed out after {timeout}s]")

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
