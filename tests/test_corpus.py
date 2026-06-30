import unittest
from pathlib import Path

from llm_routing.corpus import (
    Query,
    get_dataset,
    load_corpus,
    load_corpus_artifacts,
    partition_eval,
    partition_holdout,
    prepare_corpus,
    save_corpus,
    select_queries,
)
from llm_routing.setting import corpus_spec_from_setting, load_setting


class TestCorpus(unittest.TestCase):
    def test_arc_roundtrip(self) -> None:
        spec = get_dataset("ARC-Challenge")
        corpus = load_corpus(spec)
        self.assertEqual(len(corpus), 1471)

        holdout = partition_holdout(corpus, method="split_dataset", seed=42, selection_n=150)
        part = partition_eval(
            corpus, holdout["selection_holdout"],
            method="split_dataset", seed=42, test_n=150,
        )
        out_dir = Path("experiments/corpora/_test_arc")
        save_corpus(out_dir, corpus, {"dataset": "ARC-Challenge", "corpus_size": len(corpus)}, part)
        loaded, _, loaded_part = load_corpus_artifacts(out_dir)
        self.assertEqual(len(select_queries(loaded, loaded_part["selection_holdout"])), len(part["selection_holdout"]))

    def test_registry_splits(self) -> None:
        spec = corpus_spec_from_setting(load_setting("experiments/candidates/arc.yaml"))
        self.assertEqual(spec.corpus_splits, get_dataset("ARC-Challenge").corpus_splits)

    def test_prepare_corpus_subsample(self) -> None:
        corpus = [
            Query(
                f"MMLU_PRO:test:{i}",
                "MMLU_PRO",
                f"question {i}",
                ("a", "b"),
                0,
                {"subject": "math" if i % 2 == 0 else "law"},
            )
            for i in range(100)
        ]
        out, meta = prepare_corpus(
            corpus,
            {
                "subsample_when_above": 50,
                "target_size": 30,
                "stratify_metadata_key": "subject",
            },
            seed=0,
        )
        self.assertEqual(len(out), 30)
        self.assertTrue(meta["subsampled"])
        subjects = {q.metadata["subject"] for q in out}
        self.assertEqual(subjects, {"math", "law"})

    def test_prepare_corpus_drops_invalid(self) -> None:
        corpus = [
            Query("bad", "X", "", ("a",), 0),
            Query("ok", "X", "q", ("a", "b"), 0),
        ]
        out, meta = prepare_corpus(corpus, {}, seed=0)
        self.assertEqual(len(out), 1)
        self.assertEqual(meta["n_dropped_invalid"], 1)


if __name__ == "__main__":
    unittest.main()
