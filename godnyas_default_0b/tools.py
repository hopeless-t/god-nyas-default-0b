from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess
import time


@dataclass(frozen=True)
class ToolSpec:
    name: str
    argv: tuple[str, ...]
    cwd: str
    timeout_seconds: float = 30.0
    read_only: bool = True


@dataclass(frozen=True)
class ToolReceipt:
    name: str
    exit_code: int
    duration_ms: float
    stdout: str
    stderr: str
    stdout_sha256: str
    stderr_sha256: str
    success: bool


def run_tool(spec: ToolSpec, max_output_chars: int = 20000) -> ToolReceipt:
    cwd = Path(spec.cwd).resolve()
    if not cwd.is_dir() or not spec.argv:
        raise ValueError("invalid tool spec")
    started = time.perf_counter()
    proc = subprocess.run(list(spec.argv), cwd=str(cwd), capture_output=True, text=True, timeout=spec.timeout_seconds, shell=False, check=False)
    elapsed = (time.perf_counter() - started) * 1000
    return ToolReceipt(spec.name, proc.returncode, elapsed, proc.stdout[:max_output_chars], proc.stderr[:max_output_chars], hashlib.sha256(proc.stdout.encode()).hexdigest(), hashlib.sha256(proc.stderr.encode()).hexdigest(), proc.returncode == 0)


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = {t.name: t for t in tools}

    def execute(self, name: str) -> ToolReceipt:
        if name not in self.tools:
            raise KeyError(f"tool not allowlisted: {name}")
        return run_tool(self.tools[name])

    def manifest(self) -> list[dict]:
        return [{"name": t.name, "argv": list(t.argv), "cwd": t.cwd, "timeout_seconds": t.timeout_seconds, "read_only": t.read_only} for t in sorted(self.tools.values(), key=lambda x: x.name)]
