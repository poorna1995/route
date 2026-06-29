import unittest

from llm_routing.corpus import (
    Query,
    partition_eval,
    partition_holdout,
    resolve_holdout_size,
    resolve_test_size,
)


class TestPartition(unittest.TestCase):
    POLICY = {"test_fraction": 0.20, "test_min": 150, "test_max": 1000}

    def test_holdout_size(self) -> None:
        sel = resolve_holdout_size(1471, {"selection_holdout_n": 150})
        self.assertEqual(sel, 150)

    def test_fixed_not_fraction(self) -> None:
        for n in (817, 1172, 10042):
            sel = resolve_holdout_size(n, {"selection_holdout_n": 150})
            self.assertEqual(sel, 150)

    def test_adaptive_test_arc(self) -> None:
        # |C|=1471, |C\H|=1321 → 20% = 264
        self.assertEqual(resolve_test_size(self.POLICY, eval_pool_n=1321), 264)

    def test_adaptive_test_truthfulqa(self) -> None:
        # |C|=817, |C\H|=667 → 20% = 133 → clamp min 150
        self.assertEqual(resolve_test_size(self.POLICY, eval_pool_n=667), 150)

    def test_adaptive_test_hellaswag(self) -> None:
        # |C|=10042, |C\H|=9892 → 20% = 1978 → clamp max 1000
        self.assertEqual(resolve_test_size(self.POLICY, eval_pool_n=9892), 1000)

    def test_frozen_test_n_override(self) -> None:
        cfg = {**self.POLICY, "test_n": 150}
        self.assertEqual(resolve_test_size(cfg, eval_pool_n=9892), 150)

    def test_legacy_snapshot_m3_lock_policy(self) -> None:
        from llm_routing.setting import partition_cfg_for_m3_lock

        legacy = {
            "partition": {
                "method": "random_split",
                "seed": 42,
                "selection_holdout_n": 150,
            },
            "pool": {"M_lo": "x", "M_hi": "y"},
        }
        cfg = partition_cfg_for_m3_lock(legacy)
        self.assertEqual(resolve_test_size(cfg, eval_pool_n=1321), 264)

    def test_m1_holdout_only(self) -> None:
        corpus = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(500)]
        part = partition_holdout(corpus, method="random_split", seed=42, selection_n=150)
        self.assertEqual(len(part["selection_holdout"]), 150)
        self.assertNotIn("calib", part)

    def test_m3_extends_same_shuffle(self) -> None:
        corpus = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(500)]
        holdout = partition_holdout(corpus, method="random_split", seed=42, selection_n=150)
        test_n = resolve_test_size(self.POLICY, eval_pool_n=350)
        full = partition_eval(
            corpus, holdout["selection_holdout"],
            method="random_split", seed=42, test_n=test_n,
        )
        self.assertEqual(len(full["selection_holdout"]), 150)
        self.assertEqual(len(full["test"]), test_n)
        self.assertEqual(len(full["calib"]), 350 - test_n)

    def test_eval_query_ids(self) -> None:
        from llm_routing.corpus import eval_query_ids, validate_partition

        corpus = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(500)]
        holdout = partition_holdout(corpus, method="random_split", seed=42, selection_n=150)
        full = partition_eval(
            corpus, holdout["selection_holdout"],
            method="random_split", seed=42, test_n=150,
        )
        validate_partition(corpus, full)
        ids, calib, test = eval_query_ids(full)
        self.assertEqual(len(ids), len(calib) + len(test))
        self.assertEqual(len(calib & test), 0)
        self.assertEqual(len(set(full["selection_holdout"]) & set(ids)), 0)


if __name__ == "__main__":
    unittest.main()
