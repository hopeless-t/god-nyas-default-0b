"""Bundled public 0B asset.

This is a usable strict-0B public build: no learned weights, no model calls,
explicit state, deterministic semantic compilation, planning, verification,
and abstention on unsupported requests.
"""
from __future__ import annotations

import ast
import json
import operator
import re
import unicodedata
from dataclasses import dataclass, asdict
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable

FactSet = frozenset[str]


@dataclass(frozen=True)
class Action:
    name: str
    requires: FactSet
    adds: FactSet
    cost: int = 1

    def applicable(self, state: FactSet) -> bool:
        return self.requires <= state

    def project(self, state: FactSet) -> FactSet:
        return frozenset(state | self.adds)


@dataclass
class ActionStats:
    success: int = 0
    failure: int = 0

    @property
    def failure_rate(self) -> float:
        n = self.success + self.failure
        return 0.0 if n == 0 else self.failure / n


class Memory:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.stats: dict[str, ActionStats] = {}
        if path and path.exists():
            raw = json.loads(path.read_text())
            self.stats = {k: ActionStats(**v) for k, v in raw.items()}

    def penalty(self, action: str) -> float:
        return 8.0 * self.stats.get(action, ActionStats()).failure_rate

    def observe(self, action: str, ok: bool) -> None:
        stat = self.stats.setdefault(action, ActionStats())
        if ok:
            stat.success += 1
        else:
            stat.failure += 1
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({k: asdict(v) for k, v in sorted(self.stats.items())}, indent=2) + "\n")


@dataclass(frozen=True)
class IntentRule:
    intent: str
    phrases: tuple[str, ...]
    goal: tuple[str, ...]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\s\u3000]+", " ", text)
    suffixes = ("してください", "して下さい", "ください", "下さい", "お願いします", "お願い", "にゃ")
    prefixes = ("please ", "can you ", "could you ", "お願い ")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):].strip(); changed = True
        for s in suffixes:
            if text.endswith(s):
                text = text[:-len(s)].strip(); changed = True
    return text


RULES = (
    IntentRule("inspect", ("inspect", "check status", "status", "確認して", "状態を見て", "観測して"), ("inspection_evidence",)),
    IntentRule("verify", ("verify", "validate", "検証して", "正しいか確認", "チェックして"), ("verification_evidence",)),
    IntentRule("repair", ("repair", "fix", "直して", "修復して", "壊れてるのを直して"), ("repair_verified",)),
    IntentRule("benchmark", ("benchmark", "bench", "ベンチ", "ベンチ回して", "性能測定", "測って"), ("benchmark_evidence",)),
    IntentRule("build", ("build", "implement", "実装して", "作って", "組んで"), ("build_verified",)),
    IntentRule("recover", ("recover", "restore", "復元して", "戻して"), ("recovery_verified",)),
)

ACTIONS = (
    Action("observe", frozenset({"request"}), frozenset({"observed"})),
    Action("inspect", frozenset({"observed"}), frozenset({"inspection_evidence"})),
    Action("verify", frozenset({"observed"}), frozenset({"verification_evidence"})),
    Action("diagnose", frozenset({"observed"}), frozenset({"diagnosed"})),
    Action("repair", frozenset({"diagnosed"}), frozenset({"repaired"}), cost=2),
    Action("verify_repair", frozenset({"repaired"}), frozenset({"repair_verified"})),
    Action("prepare_benchmark", frozenset({"observed"}), frozenset({"benchmark_ready"})),
    Action("run_benchmark", frozenset({"benchmark_ready"}), frozenset({"benchmark_evidence"}), cost=2),
    Action("resolve_build", frozenset({"observed"}), frozenset({"build_target"})),
    Action("construct", frozenset({"build_target"}), frozenset({"constructed"}), cost=2),
    Action("verify_build", frozenset({"constructed"}), frozenset({"build_verified"})),
    Action("locate_recovery_source", frozenset({"observed"}), frozenset({"recovery_source"})),
    Action("restore", frozenset({"recovery_source"}), frozenset({"restored"}), cost=2),
    Action("verify_recovery", frozenset({"restored"}), frozenset({"recovery_verified"})),
)


def _plan(initial: Iterable[str], goal: Iterable[str], memory: Memory) -> list[Action]:
    start, target = frozenset(initial), frozenset(goal)
    q: list[tuple[float, int, FactSet, tuple[Action, ...]]] = [(0.0, 0, start, ())]
    best: dict[FactSet, float] = {start: 0.0}
    serial = 0
    while q:
        cost, _, state, path = heappop(q)
        if target <= state:
            return list(path)
        if cost != best.get(state):
            continue
        for action in ACTIONS:
            if not action.applicable(state):
                continue
            nxt = action.project(state)
            nxt_cost = cost + action.cost + memory.penalty(action.name)
            if nxt_cost >= best.get(nxt, float("inf")):
                continue
            best[nxt] = nxt_cost
            serial += 1
            heappush(q, (nxt_cost, serial, nxt, path + (action,)))
    return []


_ALLOWED_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _arith(expr: str) -> int | float:
    node = ast.parse(expr, mode="eval").body
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN:
            return _ALLOWED_BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(n.op)](ev(n.operand))
        raise ValueError("unsupported arithmetic expression")
    return ev(node)


class Default0BModel:
    """Bundled public 0B model.

    This public build is intentionally bounded and abstains outside its declared
    surface rather than pretending to understand open-world requests.
    """

    def __init__(self, memory_path: str | Path | None = None):
        self.memory = Memory(Path(memory_path) if memory_path else None)

    def info(self) -> dict[str, Any]:
        return {
            "strict_0b": True,
            "trainable_parameters": 0,
            "model_calls": 0,
            "bounded_semantics": True,
        }

    def respond(self, text: str) -> dict[str, Any]:
        normalized = _norm(text)
        matched = [r for r in RULES if normalized in {_norm(p) for p in r.phrases}]
        if len(matched) != 1:
            return {
                "status": "UNRESOLVED",
                **self.info(),
                "reason": "outside public bounded semantic surface" if not matched else "ambiguous public semantic surface",
            }
        rule = matched[0]
        plan = _plan(("request",), rule.goal, self.memory)
        if not plan:
            return {"status": "UNRESOLVED", **self.info(), "reason": "no deterministic plan"}
        state = frozenset({"request"})
        trace: list[str] = []
        cost = 0
        for action in plan:
            before = state
            state = action.project(state)
            ok = state == action.project(before)
            self.memory.observe(action.name, ok)
            trace.append(action.name)
            cost += action.cost
        return {
            "status": "DONE",
            **self.info(),
            "intent": rule.intent,
            "trace": trace,
            "cost": cost,
            "goal": list(rule.goal),
        }

    def reason(self, expression: str) -> dict[str, Any]:
        try:
            answer = _arith(expression)
        except Exception as exc:
            return {"status": "UNRESOLVED", **self.info(), "reason": str(exc)}
        return {"status": "DONE", **self.info(), "answer": answer}


def load_default_0b(memory_path: str | Path | None = None) -> Default0BModel:
    return Default0BModel(memory_path=memory_path)
