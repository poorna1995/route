import unittest
from pathlib import Path

from llm_routing.corpus import (
    get_dataset,
    load_corpus,
    load_corpus_artifacts,
    partition_eval,
    partition_holdout,
    save_corpus,
    select_queries,
)
from llm_routing.setting import corpus_spec_from_setting, load_setting


class TestCorpus(unittest.TestCase):
    def test_arc_roundtrip(self) -> None:
        spec = get_dataset("ARC-Challenge")
        corpus = load_corpus(spec)
        self.assertEqual(len(corpus), 1471)

        holdout = partition_holdout(corpus, method="random_split", seed=42, selection_n=150)
        part = partition_eval(
            corpus, holdout["selection_holdout"],
            method="random_split", seed=42, test_n=150,
        )
        out_dir = Path("experiments/corpora/_test_arc")
        save_corpus(out_dir, corpus, {"dataset": "ARC-Challenge", "corpus_size": len(corpus)}, part)
        loaded, _, loaded_part = load_corpus_artifacts(out_dir)
        self.assertEqual(len(select_queries(loaded, loaded_part["selection_holdout"])), len(part["selection_holdout"]))

    def test_registry_splits(self) -> None:
        spec = corpus_spec_from_setting(load_setting("experiments/candidates/arc.yaml"))
        self.assertEqual(spec.corpus_splits, get_dataset("ARC-Challenge").corpus_splits)


if __name__ == "__main__":
    unittest.main()
