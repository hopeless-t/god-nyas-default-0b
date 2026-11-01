from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable, Iterable
import json

FactSet = frozenset[str]


@dataclass(frozen=True)
class Action:
    name: str
    requires: FactSet
    adds: FactSet
    removes: FactSet = frozenset()
    cost: int = 1
    lane: str = "mechanical"

    def applicable(self, state: FactSet) -> bool:
        return self.requires <= state

    def project(self, state: FactSet) -> FactSet:
        return frozenset((state - self.removes) | self.adds)


@dataclass
class StepReceipt:
    action: str
    lane: str
    before: list[str]
    after: list[str]
    declared_after: list[str]
    matched_projection: bool
    note: str = ""


@dataclass
class RunReceipt:
    goal: list[str]
    final_state: list[str]
    reached_goal: bool
    replans: int
    steps: list[StepReceipt]
    nominal_cost: int
    model_calls: int = 0
    trainable_parameters: int = 0

    @property
    def action_trace(self) -> list[str]:
        return [step.action for step in self.steps]


@dataclass
class ActionStats:
    success: int = 0
    failure: int = 0

    @property
    def observations(self) -> int:
        return self.success + self.failure

    @property
    def failure_rate(self) -> float:
        return 0.0 if self.observations == 0 else self.failure / self.observations


class ReliabilityMemory:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.stats: dict[str, ActionStats] = {}
        if path and path.exists():
            raw = json.loads(path.read_text())
            self.stats = {name: ActionStats(**row) for name, row in raw.items()}

    def get(self, action: str) -> ActionStats:
        return self.stats.setdefault(action, ActionStats())

    def observe(self, action: str, matched: bool) -> None:
        row = self.get(action)
        if matched:
            row.success += 1
        else:
            row.failure += 1
        self.flush()

    def penalty(self, action: str, scale: float = 8.0) -> float:
        return scale * self.get(action).failure_rate

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: asdict(row) for name, row in sorted(self.stats.items())}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {"success": row.success, "failure": row.failure, "failure_rate": row.failure_rate}
            for name, row in sorted(self.stats.items())
        }


class NoPlan(RuntimeError):
    pass


def reliability_plan(
    initial: Iterable[str],
    goal: Iterable[str],
    actions: Iterable[Action],
    memory: ReliabilityMemory,
    *,
    blocked: set[str] | None = None,
    failure_penalty: float = 8.0,
) -> list[Action]:
    start = frozenset(initial)
    target = frozenset(goal)
    blocked = blocked or set()
    catalog = tuple(a for a in actions if a.name not in blocked)
    queue: list[tuple[float, int, FactSet, tuple[Action, ...]]] = [(0.0, 0, start, tuple())]
    best: dict[FactSet, float] = {start: 0.0}
    serial = 0
    while queue:
        cost, _, state, path = heappop(queue)
        if target <= state:
            return list(path)
        if cost != best.get(state):
            continue
        for action in catalog:
            if not action.applicable(state):
                continue
            nxt = action.project(state)
            edge = float(action.cost) + memory.penalty(action.name, failure_penalty)
            nxt_cost = cost + edge
            if nxt_cost >= best.get(nxt, float("inf")):
                continue
            best[nxt] = nxt_cost
            serial += 1
            heappush(queue, (nxt_cost, serial, nxt, path + (action,)))
    raise NoPlan(f"no deterministic plan from {sorted(start)} to {sorted(target)}")


class ZeroBEngine:
    def __init__(
        self,
        actions: Iterable[Action],
        executors: dict[str, Callable[[FactSet], tuple[FactSet, str]]] | None = None,
        *,
        memory: ReliabilityMemory | None = None,
        failure_penalty: float = 8.0,
    ):
        self.actions = tuple(actions)
        self.executors = dict(executors or {})
        self.memory = memory or ReliabilityMemory()
        self.failure_penalty = failure_penalty

    def run(self, initial: Iterable[str], goal: Iterable[str], max_replans: int = 16) -> RunReceipt:
        state = frozenset(initial)
        target = frozenset(goal)
        blocked: set[str] = set()
        steps: list[StepReceipt] = []
        replans = 0
        nominal_cost = 0
        while not target <= state:
            plan = reliability_plan(
                state,
                target,
                self.actions,
                self.memory,
                blocked=blocked,
                failure_penalty=self.failure_penalty,
            )
            if not plan:
                break
            drifted = False
            for action in plan:
                if not action.applicable(state):
                    replans += 1
                    drifted = True
                    break
                before = state
                declared = action.project(before)
                executor = self.executors.get(action.name)
                actual, note = executor(before) if executor else (declared, "declarative effect")
                matched = actual == declared
                self.memory.observe(action.name, matched)
                steps.append(
                    StepReceipt(
                        action=action.name,
                        lane=action.lane,
                        before=sorted(before),
                        after=sorted(actual),
                        declared_after=sorted(declared),
                        matched_projection=matched,
                        note=note,
                    )
                )
                nominal_cost += action.cost
                state = actual
                if not matched:
                    blocked.add(action.name)
                    replans += 1
                    if replans > max_replans:
                        raise NoPlan("replan budget exhausted")
                    drifted = True
                    break
                if target <= state:
                    break
            if not drifted and not target <= state:
                replans += 1
                if replans > max_replans:
                    raise NoPlan("stalled before goal")
        return RunReceipt(
            goal=sorted(target),
            final_state=sorted(state),
            reached_goal=target <= state,
            replans=replans,
            steps=steps,
            nominal_cost=nominal_cost,
        )
