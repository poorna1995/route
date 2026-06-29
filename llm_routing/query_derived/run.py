"""Stage 5 orchestration: canonicalize → extract → embed → engineer → write φ(q)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_routing.corpus import eval_query_ids, load_corpus_artifacts, select_queries, write_jsonl
from llm_routing.query_derived.config import (
    QueryDerivedRecord,
    load_query_derived_defaults,
    resolve_tokenizer_id,
)
from llm_routing.query_derived.engineer import (
    NoveltyModel,
    ZScoreModel,
    encode_canonical_texts,
    save_embedding,
)
from llm_routing.query_derived.extract import (
    TokenCounter,
    canonical_user,
    extract_ambiguity,
    extract_load,
)
from llm_routing.setting import get_protocol, load_setting


def _resolve_query_ids(
    partition: dict[str, Any] | None,
    query_ids: list[str] | None,
    *,
    allow_full_corpus: bool,
    corpus: list,
) -> tuple[list[str], set[str], set[str]]:
    if partition and partition.get("calib") and partition.get("test"):
        return eval_query_ids(partition, query_ids=query_ids)
    if allow_full_corpus:
        ids = query_ids if query_ids is not None else [q.query_id for q in corpus]
        return ids, set(), set()
    raise ValueError(
        "M3 eval partition required (calib + test). "
        "Run: python run.py lock-eval --run <run_dir> "
        "or pass allow_full_corpus=True for smoke only."
    )


def _fit_novelty(
    records: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    query_ids: list[str],
    calib_ids: set[str],
    eng_dir: Path,
    config: dict[str, Any],
) -> None:
    if not calib_ids:
        for rec in records:
            rec["novelty"] = {}
        return

    calib_matrix = [embeddings[qid] for qid in query_ids if qid in calib_ids]
    if len(calib_matrix) < 2:
        for rec in records:
            rec["novelty"] = {}
        return

    model = NoveltyModel()
    model.fit(calib_matrix, config)
    (eng_dir / "novelty_model.json").write_text(
        json.dumps(model.to_artifact(), indent=2) + "\n",
        encoding="utf-8",
    )
    for rec in records:
        rec["novelty"] = model.transform(embeddings[rec["query_id"]])


def _apply_zscore(
    records: list[dict[str, Any]],
    calib_ids: set[str],
    eng_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    z_keys = list(config.get("zscore", {}).get("continuous_keys", []))
    if not calib_ids or not z_keys:
        return records

    model = ZScoreModel()
    model.fit(records, calib_ids, z_keys)
    (eng_dir / "zscore.json").write_text(
        json.dumps(model.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return [model.transform(r) for r in records]


def run_query_derived(
    run_root: Path,
    *,
    query_ids: list[str] | None = None,
    mock_embed: bool = False,
    config: dict[str, Any] | None = None,
    allow_full_corpus: bool = False,
) -> Path:
    """Execute query-derived φ(q) pipeline for one run directory."""
    run_root = Path(run_root)
    config = config or load_query_derived_defaults()
    setting = load_setting(run_root / "setting.yaml")
    protocol = get_protocol(setting)
    corpus, _, partition = load_corpus_artifacts(run_root / "corpus")

    query_ids, calib_ids, test_ids = _resolve_query_ids(
        partition, query_ids, allow_full_corpus=allow_full_corpus, corpus=corpus
    )
    queries = select_queries(corpus, query_ids)

    tokenizer_id = resolve_tokenizer_id(setting, config)
    try:
        counter = TokenCounter(tokenizer_id)
    except Exception:
        counter = TokenCounter(None)

    signals_dir = run_root / "signals"
    emb_dir = signals_dir / "embeddings"
    eng_dir = signals_dir / "engineering"
    signals_dir.mkdir(parents=True, exist_ok=True)
    eng_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    canonical_texts: list[str] = []

    for query in queries:
        canonical = canonical_user(query, protocol)
        canonical_texts.append(canonical)
        split = "calib" if query.query_id in calib_ids else ("test" if query.query_id in test_ids else "eval")
        records.append(
            QueryDerivedRecord(
                query_id=query.query_id,
                split=split,
                load=extract_load(query, canonical, counter, config),
                ambiguity=extract_ambiguity(query),
            ).to_dict()
        )

    ecfg = config.get("embedding", {})
    model_id = ecfg.get("model_id", "sentence-transformers/all-MiniLM-L6-v2")
    dim = int(ecfg.get("dimension", 384))
    try:
        vectors = encode_canonical_texts(
            canonical_texts, model_id=model_id, mock=mock_embed, dimension=dim
        )
    except ImportError as exc:
        if not mock_embed:
            raise ImportError(
                "sentence-transformers required for embeddings; install [semantic] or pass mock_embed=True"
            ) from exc
        raise

    embeddings: dict[str, list[float]] = {}
    for query, vector in zip(queries, vectors):
        embeddings[query.query_id] = vector
        save_embedding(emb_dir / f"{query.query_id}.npy", vector)

    _fit_novelty(records, embeddings, query_ids, calib_ids, eng_dir, config)
    records = _apply_zscore(records, calib_ids, eng_dir, config)

    out_path = signals_dir / "query_derived.jsonl"
    write_jsonl(out_path, records, lambda r: r)

    meta = {
        "n_queries": len(records),
        "n_calib": sum(1 for r in records if r.get("split") == "calib"),
        "n_test": sum(1 for r in records if r.get("split") == "test"),
        "holdout_excluded": bool(calib_ids or test_ids),
        "calib_fit": bool(calib_ids),
        "mock_embed": mock_embed,
        "embedding_model": model_id if not mock_embed else "mock",
        "tokenizer_id": tokenizer_id,
        "representation": {
            "load": "jsonl.load",
            "ambiguity": "jsonl.ambiguity",
            "semantic": "embeddings/{query_id}.npy",
            "novelty": "jsonl.novelty",
        },
    }
    (signals_dir / "query_derived_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    if partition and calib_ids:
        splits = {
            "policy": "calib_union_test",
            "holdout_n": len(partition.get("selection_holdout", [])),
            "calib": partition["calib"],
            "test": partition["test"],
        }
        (signals_dir / "query_derived_splits.json").write_text(
            json.dumps(splits, indent=2) + "\n", encoding="utf-8"
        )
    return out_path
