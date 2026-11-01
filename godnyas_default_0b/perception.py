from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
import hashlib
import json


class _HTMLFacts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[dict] = []
        self._capture: dict | None = None

    def handle_starttag(self, tag: str, attrs):
        a = dict(attrs)
        item = {"tag": tag, "role": a.get("role") or tag, "id": a.get("id"), "name": a.get("name"), "type": a.get("type"), "href": a.get("href"), "aria_label": a.get("aria-label"), "text": ""}
        if tag in {"button", "a", "input", "select", "textarea", "form", "title", "h1", "h2", "h3"} or "role" in a:
            self.items.append(item)
            self._capture = item

    def handle_endtag(self, tag: str):
        self._capture = None

    def handle_data(self, data: str):
        if self._capture is not None:
            t = " ".join(data.split())
            if t:
                self._capture["text"] = (self._capture.get("text", "") + " " + t).strip()


def html_observation(html: str) -> dict:
    parser = _HTMLFacts(); parser.feed(html)
    facts = []
    for i, item in enumerate(parser.items):
        label = item.get("aria_label") or item.get("text") or item.get("name") or item.get("id") or f"item-{i}"
        facts.append({"kind": "ui", "index": i, "role": item["role"], "label": label, "attrs": {k: v for k, v in item.items() if k not in {"text", "role"} and v is not None}})
    return {"schema": "0B.Perception.v1", "source": "html", "facts": facts}


def query_ui(observation: dict, *, role: str | None = None, label_contains: str | None = None) -> list[dict]:
    needle = (label_contains or "").lower()
    return [item for item in observation.get("facts", []) if item.get("kind") == "ui" and (role is None or item.get("role") == role) and (not needle or needle in str(item.get("label", "")).lower())]


def json_observation(payload) -> dict:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"schema": "0B.Perception.v1", "source": "json", "sha256": hashlib.sha256(raw.encode()).hexdigest(), "payload": payload}


def filesystem_observation(root: Path, max_entries: int = 256) -> dict:
    root = root.resolve()
    facts = []
    for path in sorted(root.rglob("*"))[:max_entries]:
        try:
            stat = path.stat()
        except OSError:
            continue
        facts.append({"kind": "fs", "path": str(path.relative_to(root)), "type": "dir" if path.is_dir() else "file" if path.is_file() else "other", "size": stat.st_size})
    return {"schema": "0B.Perception.v1", "source": "filesystem", "root": str(root), "facts": facts}


def process_observation(records: Iterable[dict]) -> dict:
    facts = [{"kind": "process", "pid": int(r.get("pid", -1)), "name": str(r.get("name", "")), "state": str(r.get("state", "unknown")), "cpu": float(r.get("cpu", 0.0)), "rss": int(r.get("rss", 0))} for r in records]
    facts.sort(key=lambda x: (x["pid"], x["name"]))
    return {"schema": "0B.Perception.v1", "source": "process", "facts": facts}
