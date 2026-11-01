from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import re
import unicodedata


def terms(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    latin = re.findall(r"[a-z0-9_]+", text)
    jp = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]+", text)
    grams = []
    for run in jp:
        # Keep single characters for short Japanese queries, plus bigrams for
        # phrase discrimination. This stays deterministic and inspectable.
        grams.extend(list(run))
        if len(run) > 1:
            grams.extend(run[i:i+2] for i in range(len(run) - 1))
    return latin + grams


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    text: str
    source: str
    timestamp: str | None
    tags: tuple[str, ...]
    sha256: str


class MemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.records: dict[str, MemoryRecord] = {}
        if path and path.exists():
            raw = json.loads(path.read_text())
            for row in raw.get("records", []):
                rec = MemoryRecord(**{**row, "tags": tuple(row.get("tags", []))})
                self.records[rec.id] = rec

    def add(self, record_id: str, text: str, source: str = "user", timestamp: str | None = None, tags=()) -> MemoryRecord:
        rec = MemoryRecord(record_id, text, source, timestamp, tuple(sorted(set(tags))), hashlib.sha256(text.encode()).hexdigest())
        self.records[record_id] = rec
        self.flush()
        return rec

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(self.records[k]) for k in sorted(self.records)]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def search(self, query: str, k: int = 5, required_tags=()) -> list[dict]:
        qterms = terms(query)
        docs = [r for r in self.records.values() if set(required_tags) <= set(r.tags)]
        if not qterms or not docs:
            return []
        tfs = {r.id: Counter(terms(r.text + " " + " ".join(r.tags))) for r in docs}
        df = Counter()
        for tf in tfs.values():
            for term in tf:
                df[term] += 1
        avgdl = sum(sum(tf.values()) for tf in tfs.values()) / len(tfs)
        scored = []
        for rec in docs:
            tf = tfs[rec.id]
            dl = max(sum(tf.values()), 1)
            score = 0.0
            for term in qterms:
                if not tf[term]:
                    continue
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                freq = tf[term]
                score += idf * (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * dl / max(avgdl, 1)))
            if score > 0:
                scored.append((score, rec))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [{"id": r.id, "score": round(s, 6), "text": r.text, "source": r.source, "sha256": r.sha256, "tags": list(r.tags)} for s, r in scored[:k]]


def minimal_context(records: list[dict], required_terms: list[str], char_budget: int) -> dict:
    uncovered = set(terms(" ".join(required_terms)))
    selected, used = [], 0
    pool = list(records)
    while uncovered and pool:
        best_i, best = None, (-1.0, 0)
        for i, rec in enumerate(pool):
            text = rec["text"]
            if used + len(text) > char_budget:
                continue
            cover = len(uncovered & set(terms(text)))
            utility = cover / max(len(text), 1)
            if (utility, cover) > best:
                best_i, best = i, (utility, cover)
        if best_i is None or best[1] == 0:
            break
        rec = pool.pop(best_i)
        selected.append(rec)
        used += len(rec["text"])
        uncovered -= set(terms(rec["text"]))
    return {"selected": selected, "characters": used, "uncovered_terms": sorted(uncovered), "complete": not uncovered}
