from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\s\u3000]+", " ", text)
    return text


def unwrap(text: str) -> str:
    text = normalize(text)
    prefixes = ("please ", "can you ", "could you ", "お願い ")
    suffixes = (" please", "してください", "して下さい", "ください", "下さい", "お願いします", "お願い", "にゃ")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip(); changed = True
        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip(); changed = True
    return text


def ngrams(text: str, n: int = 3) -> set[str]:
    value = "^" + unwrap(text) + "$"
    return {value} if len(value) <= n else {value[i:i+n] for i in range(len(value) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / max(len(a | b), 1)


@dataclass(frozen=True)
class IntentRule:
    intent_id: str
    phrases: tuple[str, ...]
    goal: tuple[str, ...]
    seed_facts: tuple[str, ...] = ("request",)


@dataclass(frozen=True)
class CompileResult:
    status: str
    intent: dict | None
    residual: dict | None
    candidates: tuple[tuple[str, float], ...]


class SemanticCompiler:
    def __init__(self, rules: Iterable[IntentRule]):
        self.rules = tuple(rules)

    def compile(self, text: str) -> CompileResult:
        body = unwrap(text)
        matched = []
        for rule in self.rules:
            if body in {unwrap(p) for p in rule.phrases}:
                matched.append(rule)
        if len(matched) == 1:
            rule = matched[0]
            return CompileResult(
                "COMPILED",
                {"id": rule.intent_id, "facts": list(rule.seed_facts), "goal": list(rule.goal), "source": "exact-rule"},
                None,
                ((rule.intent_id, 1.0),),
            )
        if len(matched) > 1:
            return CompileResult("RESIDUAL", None, {"kind": "ambiguous"}, tuple((r.intent_id, 1.0) for r in matched))
        return CompileResult("RESIDUAL", None, {"kind": "unresolved_semantics", "text": text}, tuple())


@dataclass(frozen=True)
class Case:
    text: str
    label: str
    provenance: str


class CaseMemory:
    def __init__(self):
        self.cases: list[Case] = []

    def add(self, text: str, label: str, provenance: str = "user") -> None:
        self.cases.append(Case(text, label, provenance))

    def classify(self, text: str, threshold: float = 0.72, margin: float = 0.12) -> dict:
        query = ngrams(text)
        by_label: dict[str, list[tuple[float, Case]]] = defaultdict(list)
        for case in self.cases:
            score = jaccard(query, ngrams(case.text))
            by_label[case.label].append((score, case))
        ranked = []
        for label, rows in by_label.items():
            rows.sort(key=lambda x: (-x[0], x[1].text))
            ranked.append((rows[0][0], label, rows[0][1]))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        public = [{"label": label, "score": round(score, 6), "case": case.text} for score, label, case in ranked[:5]]
        if not ranked or ranked[0][0] < threshold:
            return {"status": "RESIDUAL", "reason": "no_similar_case", "candidates": public}
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin:
            return {"status": "RESIDUAL", "reason": "case_ambiguity", "candidates": public}
        score, label, case = ranked[0]
        return {"status": "CLASSIFIED", "label": label, "score": score, "matched_case": case.text}
