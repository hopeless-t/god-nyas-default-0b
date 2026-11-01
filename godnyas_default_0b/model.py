from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .creativity import diverse_top_k, generate_candidates
from .discovery import TransitionMiner
from .engine import Action, NoPlan, ReliabilityMemory, ZeroBEngine
from .memory import MemoryStore, minimal_context
from .perception import filesystem_observation, html_observation, json_observation, process_observation, query_ui
from .reasoning import ConstraintProblem, safe_arithmetic, shortest_path, solve_csp, solve_linear, subset_sum, topological_order
from .routing import Worker, cover_route, route
from .semantic import CaseMemory, IntentRule, SemanticCompiler
from .verification import canonical_digest, receipt as verification_receipt, verify_equal, verify_json_shape, verify_subset, verify_trace


DEFAULT_RULES = (
    IntentRule("inspect", ("inspect", "check status", "status", "確認して", "状態を見て", "観測して"), ("inspection_evidence",)),
    IntentRule("verify", ("verify", "validate", "検証して", "正しいか確認", "チェックして"), ("verification_evidence",)),
    IntentRule("repair", ("repair", "fix", "直して", "修復して", "壊れてるのを直して"), ("repair_verified",)),
    IntentRule("benchmark", ("benchmark", "bench", "ベンチ", "ベンチ回して", "性能測定", "測って"), ("benchmark_evidence",)),
    IntentRule("build", ("build", "implement", "実装して", "作って", "組んで"), ("build_verified",)),
    IntentRule("recover", ("recover", "restore", "復元して", "戻して"), ("recovery_verified",)),
)

DEFAULT_ACTIONS = (
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


class Default0BModel:
    """Usable public strict-0B agent substrate.

    0B means this asset performs its declared work with zero trainable model
    parameters and zero model calls. Python is only the implementation language.
    The semantic surface is bounded; unsupported open-world requests abstain.
    """

    def __init__(self, state_dir: str | Path | None = None):
        root = Path(state_dir).expanduser() if state_dir else None
        reliability_path = root / "reliability.json" if root else None
        memory_path = root / "memory.json" if root else None
        self.rules = DEFAULT_RULES
        self.compiler = SemanticCompiler(self.rules)
        self.cases = CaseMemory()
        for rule in self.rules:
            for phrase in rule.phrases:
                self.cases.add(phrase, rule.intent_id, "default")
        self.reliability = ReliabilityMemory(reliability_path)
        self.engine = ZeroBEngine(DEFAULT_ACTIONS, memory=self.reliability)
        self.memory = MemoryStore(memory_path)
        self.discovery = TransitionMiner(min_support=2)

    def info(self) -> dict[str, Any]:
        return {
            "name": "God Nyazu default 0B public asset",
            "strict_0b": True,
            "trainable_parameters": 0,
            "model_calls": 0,
            "implementation": "deterministic software",
            "bounded_semantics": True,
            "capabilities": self.capabilities(),
        }

    def capabilities(self) -> list[str]:
        return [
            "bounded_command_understanding",
            "explicit_abstention",
            "state_space_planning",
            "reliability_weighted_replanning",
            "nonparametric_case_memory",
            "persistent_text_memory",
            "deterministic_retrieval",
            "safe_arithmetic",
            "linear_equation_solving",
            "shortest_path",
            "subset_sum",
            "topological_sort",
            "constraint_solving",
            "html_perception",
            "json_perception",
            "filesystem_perception",
            "process_perception",
            "transition_discovery",
            "deterministic_routing",
            "multi_worker_cover_routing",
            "deterministic_verification",
            "combinatorial_candidate_generation",
            "allowlisted_tool_execution",
            "checkpointed_workflows",
        ]

    def respond(self, text: str) -> dict[str, Any]:
        compiled = self.compiler.compile(text)
        intent = compiled.intent
        semantic_source = "exact-rule"
        if intent is None:
            case = self.cases.classify(text, threshold=0.55, margin=0.08)
            if case.get("status") == "CLASSIFIED":
                rule = next((r for r in self.rules if r.intent_id == case["label"]), None)
                if rule is not None:
                    intent = {"id": rule.intent_id, "facts": list(rule.seed_facts), "goal": list(rule.goal), "source": "case-memory"}
                    semantic_source = "case-memory"
        if intent is None:
            return {
                "status": "UNRESOLVED",
                "strict_0b": True,
                "model_calls": 0,
                "trainable_parameters": 0,
                "reason": "outside bounded semantic surface",
                "candidates": list(compiled.candidates),
            }
        try:
            run = self.engine.run(intent["facts"], intent["goal"])
        except NoPlan as exc:
            return {"status": "UNRESOLVED", "strict_0b": True, "model_calls": 0, "trainable_parameters": 0, "reason": str(exc)}
        return {
            "status": "DONE" if run.reached_goal else "FAILED",
            "strict_0b": True,
            "model_calls": 0,
            "trainable_parameters": 0,
            "intent": intent["id"],
            "semantic_source": semantic_source,
            "trace": run.action_trace,
            "replans": run.replans,
            "cost": run.nominal_cost,
            "final_state": run.final_state,
        }

    def reason(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if kind == "arithmetic":
                answer = safe_arithmetic(str(payload["expr"]))
            elif kind == "linear":
                answer = str(solve_linear(int(payload["a"]), int(payload["b"]), int(payload["c"])))
            elif kind == "shortest_path":
                answer = shortest_path(payload["graph"], payload["start"], payload["goal"])
            elif kind == "subset_sum":
                answer = subset_sum([int(v) for v in payload["values"]], int(payload["target"]))
            elif kind == "topological_sort":
                answer = topological_order(payload["edges"])
            elif kind == "csp":
                constraints = tuple(payload["constraints"])
                problem = ConstraintProblem({k: tuple(v) for k, v in payload["domains"].items()}, constraints)
                answer = solve_csp(problem)
            else:
                return {"status": "UNRESOLVED", "strict_0b": True, "model_calls": 0, "trainable_parameters": 0, "reason": "unsupported reasoning kernel"}
        except Exception as exc:
            return {"status": "UNRESOLVED", "strict_0b": True, "model_calls": 0, "trainable_parameters": 0, "reason": str(exc)}
        return {"status": "DONE", "answer": answer, "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def remember(self, record_id: str, text: str, source: str = "user", tags: Iterable[str] = ()) -> dict:
        rec = self.memory.add(record_id, text, source=source, tags=tuple(tags))
        return {"status": "DONE", "record": rec.__dict__, "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def recall(self, query: str, k: int = 5, required_tags: Iterable[str] = ()) -> dict:
        rows = self.memory.search(query, k=k, required_tags=tuple(required_tags))
        return {"status": "DONE", "results": rows, "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def context(self, query: str, required_terms: list[str], char_budget: int = 2000, k: int = 20) -> dict:
        rows = self.memory.search(query, k=k)
        return {"status": "DONE", **minimal_context(rows, required_terms, char_budget), "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def perceive_html(self, html: str, role: str | None = None, label_contains: str | None = None) -> dict:
        observation = html_observation(html)
        return {"status": "DONE", "observation": observation, "matches": query_ui(observation, role=role, label_contains=label_contains), "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def perceive_json(self, payload: Any) -> dict:
        return {"status": "DONE", "observation": json_observation(payload), "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def perceive_filesystem(self, root: str | Path, max_entries: int = 256) -> dict:
        return {"status": "DONE", "observation": filesystem_observation(Path(root), max_entries=max_entries), "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def perceive_processes(self, records: Iterable[dict]) -> dict:
        return {"status": "DONE", "observation": process_observation(records), "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def discover_transition(self, label: str, before: Iterable[str], after: Iterable[str]) -> dict:
        self.discovery.observe(label, before, after)
        return {"status": "DONE", "inference": self.discovery.infer(label), "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def route(self, required: Iterable[str], workers: Iterable[dict], cover: bool = False) -> dict:
        pool = [Worker(name=w["name"], capabilities=frozenset(w.get("capabilities", ())), load=float(w.get("load", 0.0)), healthy=bool(w.get("healthy", True)), lease_epoch=int(w.get("lease_epoch", 0)), cost=float(w.get("cost", 1.0))) for w in workers]
        return cover_route(set(required), pool) if cover else route(set(required), pool)

    def verify(self, observed: Any, expected: Any = None, *, subset: bool = False, required_paths: list[str] | None = None, trace_order: list[str] | None = None, forbidden: list[str] | None = None) -> dict:
        checks = []
        if required_paths is not None:
            checks.extend(verify_json_shape(observed, required_paths))
        elif trace_order is not None or forbidden is not None:
            checks.extend(verify_trace(list(observed), trace_order or [], forbidden or []))
        elif subset:
            checks.append(verify_subset("subset", observed, expected))
        else:
            checks.append(verify_equal("equal", observed, expected))
        return verification_receipt(checks)

    def digest(self, payload: Any) -> str:
        return canonical_digest(payload)

    def ideate(self, dimensions: dict[str, Iterable[str]], feature_fn, weights: dict[str, float], constraints=(), k: int = 5) -> dict:
        candidates = generate_candidates(dimensions, feature_fn, weights, constraints)
        selected = diverse_top_k(candidates, k)
        return {"status": "DONE", "generated": len(candidates), "selected": [c.as_dict() for c in selected], "strict_0b": True, "model_calls": 0, "trainable_parameters": 0}

    def self_test(self) -> dict[str, Any]:
        checks: dict[str, bool] = {}
        checks["command"] = self.respond("ベンチ回して").get("status") == "DONE"
        checks["unknown_abstains"] = self.respond("未知の宇宙法則を作って").get("status") == "UNRESOLVED"
        checks["arithmetic"] = self.reason("arithmetic", {"expr": "(12+5)*3"}).get("answer") == 51
        checks["linear"] = self.reason("linear", {"a": 2, "b": 4, "c": 10}).get("answer") == "3"
        checks["subset_sum"] = self.reason("subset_sum", {"values": [3, 7, 11], "target": 10}).get("answer") == [3, 7]
        checks["html"] = len(self.perceive_html('<button aria-label="Save">x</button>', role="button", label_contains="save")["matches"]) == 1
        self.discover_transition("demo", ["a"], ["a", "b"])
        checks["discovery"] = self.discover_transition("demo", ["a"], ["a", "b"])["inference"].get("status") == "QUALIFIED"
        checks["verification"] = self.verify({"a": {"b": 1}}, required_paths=["a.b"])["passed"]
        route_result = self.route(["x"], [{"name": "w1", "capabilities": ["x"], "load": 0.1}])
        checks["routing"] = route_result.get("worker") == "w1"
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "strict_0b": True,
            "model_calls": 0,
            "trainable_parameters": 0,
        }


def load_default_0b(state_dir: str | Path | None = None, **kwargs) -> Default0BModel:
    # memory_path is accepted for compatibility with the first public build.
    if state_dir is None and "memory_path" in kwargs:
        memory_path = kwargs.get("memory_path")
        if memory_path:
            state_dir = Path(memory_path).expanduser().parent
    return Default0BModel(state_dir=state_dir)
