from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from typing import Callable, Iterable
import ast
import operator

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_arithmetic(expr: str):
    node = ast.parse(expr, mode="eval").body
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](ev(n.operand))
        raise ValueError("unsupported arithmetic expression")
    return ev(node)


def solve_linear(a: int, b: int, c: int) -> Fraction:
    if a == 0:
        raise ValueError("a must be non-zero")
    return Fraction(c - b, a)


def shortest_path(graph: dict[str, dict[str, float]], start: str, goal: str) -> dict:
    q = [(0.0, start, (start,))]
    best = {start: 0.0}
    while q:
        cost, node, path = heappop(q)
        if node == goal:
            return {"cost": cost, "path": list(path)}
        if cost != best.get(node):
            continue
        for nxt, weight in sorted(graph.get(node, {}).items()):
            nc = cost + float(weight)
            if nc < best.get(nxt, float("inf")):
                best[nxt] = nc
                heappush(q, (nc, nxt, path + (nxt,)))
    raise ValueError("no path")


def subset_sum(values: list[int], target: int) -> list[int] | None:
    reachable: dict[int, tuple[int, ...]] = {0: tuple()}
    for value in values:
        snapshot = list(reachable.items())
        for total, chosen in snapshot:
            nxt = total + value
            if nxt not in reachable:
                reachable[nxt] = chosen + (value,)
            if nxt == target:
                return list(reachable[nxt])
    return list(reachable[target]) if target in reachable else None


def topological_order(edges: dict[str, Iterable[str]]) -> list[str]:
    nodes = set(edges)
    for values in edges.values():
        nodes.update(values)
    indegree = {n: 0 for n in nodes}
    outgoing = {n: set() for n in nodes}
    for src, dsts in edges.items():
        for dst in dsts:
            if dst not in outgoing[src]:
                outgoing[src].add(dst)
                indegree[dst] += 1
    q = deque(sorted(n for n, d in indegree.items() if d == 0))
    out = []
    while q:
        node = q.popleft()
        out.append(node)
        for dst in sorted(outgoing[node]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                q.append(dst)
    if len(out) != len(nodes):
        raise ValueError("cycle detected")
    return out


@dataclass(frozen=True)
class ConstraintProblem:
    domains: dict[str, tuple[object, ...]]
    constraints: tuple[Callable[[dict[str, object]], bool], ...]


def solve_csp(problem: ConstraintProblem) -> dict[str, object] | None:
    keys = sorted(problem.domains)
    assignment: dict[str, object] = {}
    def search(i: int):
        if i == len(keys):
            return dict(assignment)
        key = keys[i]
        for value in problem.domains[key]:
            assignment[key] = value
            if all(fn(assignment) for fn in problem.constraints):
                solved = search(i + 1)
                if solved is not None:
                    return solved
        assignment.pop(key, None)
        return None
    return search(0)
