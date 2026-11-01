from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable
from .engine import Action


@dataclass(frozen=True)
class TransitionEvidence:
    label: str
    before: frozenset[str]
    after: frozenset[str]

    @property
    def adds(self) -> frozenset[str]:
        return self.after - self.before

    @property
    def removes(self) -> frozenset[str]:
        return self.before - self.after


class TransitionMiner:
    def __init__(self, min_support: int = 2):
        self.min_support = min_support
        self._obs: dict[str, list[TransitionEvidence]] = defaultdict(list)

    def observe(self, label: str, before: Iterable[str], after: Iterable[str]) -> None:
        self._obs[label].append(TransitionEvidence(label, frozenset(before), frozenset(after)))

    def infer(self, label: str) -> dict:
        rows = self._obs.get(label, [])
        if len(rows) < self.min_support:
            return {"status": "INSUFFICIENT_EVIDENCE", "support": len(rows)}
        signatures = Counter((tuple(sorted(r.adds)), tuple(sorted(r.removes))) for r in rows)
        (adds, removes), support = signatures.most_common(1)[0]
        conflicts = len(rows) - support
        if conflicts:
            return {"status": "CONFLICT", "support": support, "conflicts": conflicts, "candidate_adds": list(adds), "candidate_removes": list(removes)}
        common_before = set(rows[0].before)
        for row in rows[1:]:
            common_before &= set(row.before)
        return {"status": "QUALIFIED", "support": support, "requires": sorted(common_before), "adds": list(adds), "removes": list(removes)}

    def promote(self, label: str, cost: int = 1) -> Action | None:
        row = self.infer(label)
        if row["status"] != "QUALIFIED":
            return None
        return Action(
            name=f"discovered:{label}",
            requires=frozenset(row["requires"]),
            adds=frozenset(row["adds"]),
            removes=frozenset(row["removes"]),
            cost=cost,
            lane="discovered",
        )


def capability_closure(seed: Iterable[str], actions: Iterable[Action], max_rounds: int = 128) -> set[str]:
    facts = set(seed)
    for _ in range(max_rounds):
        before = set(facts)
        for action in actions:
            if action.requires <= facts:
                facts -= set(action.removes)
                facts |= set(action.adds)
        if facts == before:
            break
    return facts


def shortest_capability_chain(seed: Iterable[str], goal: Iterable[str], actions: Iterable[Action]) -> list[str] | None:
    start, target = frozenset(seed), frozenset(goal)
    q = deque([(start, tuple())])
    seen = {start}
    catalog = tuple(actions)
    while q:
        state, path = q.popleft()
        if target <= state:
            return list(path)
        for action in catalog:
            if not action.requires <= state:
                continue
            nxt = action.project(state)
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + (action.name,)))
    return None
