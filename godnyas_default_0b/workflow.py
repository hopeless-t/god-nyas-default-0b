from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from .tools import ToolRegistry


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    tool: str
    requires_success_of: tuple[str, ...] = ()


class WorkflowLedger:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"completed": {}, "failed": {}}
        if path.exists():
            self.data = json.loads(path.read_text())

    def completed(self, step_id: str) -> bool:
        return step_id in self.data.get("completed", {})

    def success(self, step_id: str) -> bool:
        return self.data.get("completed", {}).get(step_id, {}).get("success", False)

    def record(self, step: WorkflowStep, receipt) -> None:
        bucket = "completed" if receipt.success else "failed"
        self.data.setdefault(bucket, {})[step.id] = {"tool": step.tool, "success": receipt.success, "exit_code": receipt.exit_code, "duration_ms": receipt.duration_ms, "stdout_sha256": receipt.stdout_sha256, "stderr_sha256": receipt.stderr_sha256}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")


class Workflow:
    def __init__(self, registry: ToolRegistry, steps: list[WorkflowStep], ledger: WorkflowLedger):
        self.registry, self.steps, self.ledger = registry, steps, ledger

    def run(self, stop_after: int | None = None) -> dict:
        executed, skipped, blocked = [], [], []
        newly_executed = 0
        for step in self.steps:
            if self.ledger.completed(step.id):
                skipped.append(step.id); continue
            if not all(self.ledger.success(dep) for dep in step.requires_success_of):
                blocked.append(step.id); continue
            receipt = self.registry.execute(step.tool)
            self.ledger.record(step, receipt)
            executed.append({"step": step.id, "tool": step.tool, "success": receipt.success, "exit_code": receipt.exit_code})
            newly_executed += 1
            if not receipt.success or (stop_after is not None and newly_executed >= stop_after):
                break
        all_done = all(self.ledger.completed(step.id) and self.ledger.success(step.id) for step in self.steps)
        return {"all_done": all_done, "executed": executed, "skipped": skipped, "blocked": blocked, "model_calls": 0, "trainable_parameters": 0}
