import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_routing.corpus import Query, QueryResult
from llm_routing.run import (
    ROOT,
    Run,
    build_selection_report,
    compute_scorecard,
    pick_winner,
    stage_selection_report,
)
from llm_routing.setting import load_setting


class TestPipeline(unittest.TestCase):
    def test_scorecard(self) -> None:
        queries = [Query("q1", "ARC", "t", ("a", "b"), 0), Query("q2", "ARC", "t", ("a", "b"), 1)]
        lo = [QueryResult("q1", "lo", "A", 0, 1), QueryResult("q2", "lo", "A", 0, 0)]
        hi = [QueryResult("q1", "hi", "A", 0, 1), QueryResult("q2", "hi", "B", 1, 1)]
        gates = {
            "min_accuracy_gap": 0.03,
            "opportunity_min": 0.05,
            "too_hard_max": 0.70,
        }
        s = compute_scorecard(queries, lo, hi, gates)
        self.assertTrue(s["gate_c_pass"])
        self.assertEqual(s["buckets"]["opportunity"], 1)

    def test_gate_c_min_accuracy_gap(self) -> None:
        queries = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(100)]
        # 52% lo, 53% hi → gap 0.01 < 0.03
        lo = [QueryResult(f"q{i}", "lo", "A", 0, int(i < 52)) for i in range(100)]
        hi = [QueryResult(f"q{i}", "hi", "A", 0, int(i < 53)) for i in range(100)]
        gates = {
            "min_accuracy_gap": 0.03,
            "opportunity_min": 0.05,
            "too_hard_max": 0.70,
        }
        s = compute_scorecard(queries, lo, hi, gates)
        self.assertFalse(s["gate_c_pass"])
        self.assertAlmostEqual(s["acc_gap"], 0.01)

    def test_gate_d_high_opportunity_passes(self) -> None:
        """70% opportunity is valid when too_hard stays low — no upper bound on opportunity."""
        n = 100
        queries = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(n)]
        # 15 easy, 70 opportunity, 10 lo_only, 5 too_hard
        lo = []
        hi = []
        for i in range(n):
            if i < 15:
                lo_ok, hi_ok = 1, 1
            elif i < 85:
                lo_ok, hi_ok = 0, 1
            elif i < 95:
                lo_ok, hi_ok = 1, 0
            else:
                lo_ok, hi_ok = 0, 0
            lo.append(QueryResult(f"q{i}", "lo", "A", 0, lo_ok))
            hi.append(QueryResult(f"q{i}", "hi", "A", 0, hi_ok))
        gates = {
            "min_accuracy_gap": 0.03,
            "opportunity_min": 0.05,
            "too_hard_max": 0.70,
        }
        s = compute_scorecard(queries, lo, hi, gates)
        self.assertAlmostEqual(s["opportunity_rate"], 0.70)
        self.assertTrue(s["gate_d_pass"])

    def test_run_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("llm_routing.paths.RUNS_ROOT", Path(tmp)):
                run = Run.create(ROOT / "experiments/candidates/arc.yaml", name="test")
                self.assertTrue((run.root / "setting.yaml").exists())
                self.assertIn("setting_source", run.manifest())
                setting = load_setting(run.setting_path)
                self.assertIn("pool", setting)
                self.assertEqual(setting["corpus"]["dataset"], "ARC-Challenge")

    def test_phase_a_defaults_merge(self) -> None:
        setting = load_setting(ROOT / "experiments/candidates/arc.yaml")
        self.assertEqual(setting["partition"]["selection_holdout_n"], 150)
        self.assertEqual(setting["partition"]["test_fraction"], 0.20)
        self.assertEqual(setting["partition"]["test_min"], 150)
        self.assertEqual(setting["partition"]["test_max"], 1000)
        self.assertIn("selection", setting)

    def test_selection_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            run_dir = runs / "20260101-arc"
            run_dir.mkdir()
            sc = {
                "run_id": "20260101-arc",
                "dataset": "ARC-Challenge",
                "acc_lo": 0.4, "acc_hi": 0.6, "acc_gap": 0.2,
                "opportunity_rate": 0.25, "too_hard_rate": 0.3,
                "gate_c_pass": True, "gate_d_pass": True,
                "gates": {
                    "min_accuracy_gap": 0.03,
                    "opportunity_min": 0.05,
                    "too_hard_max": 0.70,
                },
                "buckets": {"opportunity": 38}, "n": 150,
            }
            (run_dir / "scorecard.json").write_text(json.dumps(sc))
            (run_dir / "manifest.json").write_text("{}")
            (run_dir / "setting.yaml").write_text(
                (ROOT / "experiments/candidates/arc.yaml").read_text()
            )
            report = build_selection_report(runs_root=runs)
            self.assertEqual(report["winner"], "ARC-Challenge")
            self.assertEqual(len(report["candidates"]), 1)
            self.assertTrue(report["candidates"][0]["passed"])

    def test_pick_winner_tie_break(self) -> None:
        cands = [
            {"candidate": "A", "passed": True, "scorecard": {"opportunity_rate": 0.2, "acc_gap": 0.1}},
            {"candidate": "B", "passed": True, "scorecard": {"opportunity_rate": 0.3, "acc_gap": 0.05}},
        ]
        tie = {"order": ["acc_gap", "opportunity_rate"], "prefer": []}
        self.assertEqual(pick_winner(cands, tie), "A")


if __name__ == "__main__":
    unittest.main()
