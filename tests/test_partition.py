import unittest

from llm_routing.corpus import (
    Query,
    partition_eval,
    partition_holdout,
    resolve_holdout_size,
    resolve_test_size,
)


class TestPartition(unittest.TestCase):
    def test_holdout_size(self) -> None:
        sel = resolve_holdout_size(1471, {"selection_holdout_n": 150})
        self.assertEqual(sel, 150)

    def test_fixed_not_fraction(self) -> None:
        # Same holdout count regardless of corpus size
        for n in (817, 1172, 10042):
            sel = resolve_holdout_size(n, {"selection_holdout_n": 150})
            self.assertEqual(sel, 150)

    def test_m1_holdout_only(self) -> None:
        corpus = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(500)]
        part = partition_holdout(corpus, method="random_split", seed=42, selection_n=150)
        self.assertEqual(len(part["selection_holdout"]), 150)
        self.assertNotIn("calib", part)

    def test_m3_extends_same_shuffle(self) -> None:
        corpus = [Query(f"q{i}", "ARC", "t", ("a", "b"), 0) for i in range(500)]
        holdout = partition_holdout(corpus, method="random_split", seed=42, selection_n=150)
        full = partition_eval(
            corpus, holdout["selection_holdout"],
            method="random_split", seed=42, test_n=150,
        )
        self.assertEqual(len(full["selection_holdout"]), 150)
        self.assertEqual(len(full["test"]), 150)
        self.assertEqual(len(full["calib"]), 200)

    def test_test_n(self) -> None:
        self.assertEqual(resolve_test_size({"test_n": 150}), 150)


if __name__ == "__main__":
    unittest.main()
