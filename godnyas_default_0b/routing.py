from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    name: str
    capabilities: frozenset[str]
    load: float = 0.0
    healthy: bool = True
    lease_epoch: int = 0
    cost: float = 1.0


def route(required: set[str], workers: list[Worker], *, max_load: float = 0.95, min_lease_epoch: int = 0) -> dict:
    eligible, rejected = [], {}
    for worker in workers:
        reasons = []
        if not worker.healthy:
            reasons.append("unhealthy")
        if worker.load > max_load:
            reasons.append("overloaded")
        if worker.lease_epoch < min_lease_epoch:
            reasons.append("stale_lease")
        missing = required - set(worker.capabilities)
        if missing:
            reasons.append("missing:" + ",".join(sorted(missing)))
        if reasons:
            rejected[worker.name] = reasons
        else:
            eligible.append(worker)
    if not eligible:
        return {"status": "NO_ROUTE", "required": sorted(required), "rejected": rejected, "model_calls": 0, "trainable_parameters": 0}
    eligible.sort(key=lambda w: (w.load, w.cost, w.name))
    chosen = eligible[0]
    return {"status": "ROUTED", "worker": chosen.name, "required": sorted(required), "load": chosen.load, "cost": chosen.cost, "rejected": rejected, "model_calls": 0, "trainable_parameters": 0}


def cover_route(required: set[str], workers: list[Worker], *, max_workers: int = 4) -> dict:
    remaining, selected = set(required), []
    pool = [w for w in workers if w.healthy and w.load <= 0.95]
    while remaining and pool and len(selected) < max_workers:
        pool.sort(key=lambda w: (-len(remaining & set(w.capabilities)), w.load, w.cost, w.name))
        best = pool.pop(0)
        cover = remaining & set(best.capabilities)
        if not cover:
            break
        selected.append({"worker": best.name, "covers": sorted(cover)})
        remaining -= cover
    return {"status": "ROUTED" if not remaining else "RESIDUAL", "selected": selected, "uncovered": sorted(remaining), "model_calls": 0, "trainable_parameters": 0}
