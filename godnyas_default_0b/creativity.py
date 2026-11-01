from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable


@dataclass(frozen=True)
class Candidate:
    choices: tuple[tuple[str, str], ...]
    features: tuple[tuple[str, float], ...]
    score: float

    def as_dict(self) -> dict:
        return {"choices": dict(self.choices), "features": dict(self.features), "score": self.score}


def generate_candidates(dimensions: dict[str, Iterable[str]], feature_fn: Callable[[dict[str, str]], dict[str, float]], weights: dict[str, float], constraints: Iterable[Callable[[dict[str, str]], bool]] = ()) -> list[Candidate]:
    keys = tuple(sorted(dimensions))
    values = [tuple(dimensions[k]) for k in keys]
    out = []
    for combo in product(*values):
        choices = dict(zip(keys, combo))
        if not all(fn(choices) for fn in constraints):
            continue
        features = feature_fn(choices)
        score = sum(features.get(name, 0.0) * weight for name, weight in weights.items())
        out.append(Candidate(tuple(sorted(choices.items())), tuple(sorted(features.items())), score))
    return sorted(out, key=lambda c: (-c.score, c.choices))


def diverse_top_k(candidates: Iterable[Candidate], k: int, diversity_weight: float = 0.25) -> list[Candidate]:
    pool = list(candidates)
    if not pool or k <= 0:
        return []
    selected = [pool.pop(0)]
    while pool and len(selected) < k:
        def distance(a, b):
            aa, bb = set(a.choices), set(b.choices)
            return 1.0 - len(aa & bb) / max(len(aa | bb), 1)
        best_i, best_value = 0, float("-inf")
        for i, cand in enumerate(pool):
            novelty = min(distance(cand, chosen) for chosen in selected)
            value = cand.score + diversity_weight * novelty
            if value > best_value:
                best_i, best_value = i, value
        selected.append(pool.pop(best_i))
    return selected
