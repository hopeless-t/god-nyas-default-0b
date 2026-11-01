from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    observed: object
    expected: object


def verify_equal(name: str, observed, expected) -> Check:
    return Check(name, observed == expected, observed, expected)


def verify_subset(name: str, observed, expected_subset) -> Check:
    return Check(name, set(expected_subset) <= set(observed), list(observed), list(expected_subset))


def verify_file_hash(path: Path, expected_sha256: str) -> Check:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    return Check("sha256", got == expected_sha256, got, expected_sha256)


def verify_json_shape(payload, required_paths: list[str]) -> list[Check]:
    checks = []
    for dotted in required_paths:
        cur, ok = payload, True
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok, cur = False, None
                break
        checks.append(Check(f"json:{dotted}", ok, cur, "present"))
    return checks


def verify_trace(trace: list[str], required_order: list[str] = (), forbidden: list[str] = ()) -> list[Check]:
    checks = []
    if required_order:
        cursor, positions, ok = -1, [], True
        for item in required_order:
            try:
                cursor = trace.index(item, cursor + 1); positions.append(cursor)
            except ValueError:
                ok = False; break
        checks.append(Check("trace_order", ok, positions, required_order))
    for item in forbidden:
        checks.append(Check(f"trace_forbidden:{item}", item not in trace, item in trace, False))
    return checks


def receipt(checks: list[Check]) -> dict:
    return {"passed": all(c.passed for c in checks), "checks": [c.__dict__ for c in checks], "model_calls": 0, "trainable_parameters": 0}


def canonical_digest(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
