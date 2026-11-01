import tempfile
import unittest
from pathlib import Path

from godnyas_default_0b import Action, ReliabilityMemory, ZeroBEngine, load_default_0b
from godnyas_default_0b.reasoning import ConstraintProblem, solve_csp, topological_order
from godnyas_default_0b.tools import ToolRegistry, ToolSpec
from godnyas_default_0b.workflow import Workflow, WorkflowLedger, WorkflowStep


class DefaultAssetTest(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(load_default_0b().self_test()["passed"])

    def test_command_planning(self):
        result = load_default_0b().respond("ベンチ回して")
        self.assertEqual(result["status"], "DONE")
        self.assertIn("run_benchmark", result["trace"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["trainable_parameters"], 0)

    def test_unknown_abstains(self):
        self.assertEqual(load_default_0b().respond("未知の宇宙法則を作って")["status"], "UNRESOLVED")

    def test_replan_after_drift(self):
        actions = (
            Action("start", frozenset({"a"}), frozenset({"b"})),
            Action("fast", frozenset({"b"}), frozenset({"done"})),
            Action("repair", frozenset({"failed"}), frozenset({"done"})),
        )
        fired = {"x": False}
        def fast(state):
            if not fired["x"]:
                fired["x"] = True
                return frozenset((state - {"done"}) | {"failed"}), "observed mismatch"
            return frozenset(state | {"done"}), "ok"
        run = ZeroBEngine(actions, {"fast": fast}, memory=ReliabilityMemory()).run(["a"], ["done"])
        self.assertTrue(run.reached_goal)
        self.assertGreaterEqual(run.replans, 1)
        self.assertIn("repair", run.action_trace)

    def test_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            model = load_default_0b(Path(td))
            model.remember("1", "東京の猫は屋根で寝ている", tags=["cat"])
            rows = model.recall("猫", required_tags=["cat"])["results"]
            self.assertEqual(rows[0]["id"], "1")

    def test_reasoning_kernels(self):
        model = load_default_0b()
        self.assertEqual(model.reason("arithmetic", {"expr": "(12+5)*3"})["answer"], 51)
        self.assertEqual(model.reason("linear", {"a": 2, "b": 4, "c": 10})["answer"], "3")
        self.assertEqual(model.reason("subset_sum", {"values": [3, 7, 11], "target": 10})["answer"], [3, 7])
        path = model.reason("shortest_path", {"graph": {"a": {"b": 2, "c": 5}, "b": {"c": 1}, "c": {}}, "start": "a", "goal": "c"})["answer"]
        self.assertEqual(path, {"cost": 3.0, "path": ["a", "b", "c"]})

    def test_topological_and_csp(self):
        self.assertEqual(topological_order({"a": ["b"], "b": ["c"]}), ["a", "b", "c"])
        problem = ConstraintProblem({"x": (1, 2), "y": (1, 2)}, (lambda a: "x" not in a or "y" not in a or a["x"] != a["y"],))
        solved = solve_csp(problem)
        self.assertNotEqual(solved["x"], solved["y"])

    def test_perception(self):
        model = load_default_0b()
        result = model.perceive_html('<button aria-label="Save">Go</button>', role="button", label_contains="save")
        self.assertEqual(len(result["matches"]), 1)
        obs = model.perceive_json({"x": 1})["observation"]
        self.assertEqual(obs["source"], "json")

    def test_discovery_requires_repeat(self):
        model = load_default_0b()
        a = model.discover_transition("x", ["a"], ["a", "b"])["inference"]
        b = model.discover_transition("x", ["a"], ["a", "b"])["inference"]
        self.assertEqual(a["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(b["status"], "QUALIFIED")

    def test_routing(self):
        model = load_default_0b()
        workers = [
            {"name": "busy", "capabilities": ["a"], "load": 0.99},
            {"name": "ok", "capabilities": ["a", "b"], "load": 0.2},
        ]
        self.assertEqual(model.route(["a"], workers)["worker"], "ok")
        self.assertEqual(model.route(["a", "b"], workers, cover=True)["status"], "ROUTED")

    def test_verification(self):
        model = load_default_0b()
        self.assertTrue(model.verify({"a": {"b": 1}}, required_paths=["a.b"])["passed"])
        self.assertFalse(model.verify(["a"], ["a", "b"], subset=True)["passed"])

    def test_ideation(self):
        model = load_default_0b()
        result = model.ideate(
            {"shape": ["round", "square"], "size": ["s", "l"]},
            lambda c: {"utility": (2 if c["size"] == "l" else 1) + (1 if c["shape"] == "round" else 0)},
            {"utility": 1.0},
            k=2,
        )
        self.assertEqual(result["generated"], 4)
        self.assertEqual(len(result["selected"]), 2)

    def test_allowlisted_tool_and_checkpointed_workflow(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ToolRegistry([ToolSpec("echo", ("python3", "-c", "print('nya')"), td)])
            ledger = WorkflowLedger(Path(td) / "ledger.json")
            flow = Workflow(registry, [WorkflowStep("one", "echo")], ledger)
            self.assertTrue(flow.run()["all_done"])
            self.assertEqual(flow.run()["executed"], [])


if __name__ == "__main__":
    unittest.main()
