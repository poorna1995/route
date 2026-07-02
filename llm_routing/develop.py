""" Development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_routing.corpus import (
    Query,
    QueryResult,
    eval_query_ids,
    load_corpus_artifacts,
    read_jsonl,
    select_split,
    write_jsonl,
)
from llm_routing.run import Run, write_json
from llm_routing.setting import get_protocol
from llm_routing.signals.psi.protocol import ARTIFACT_VERSION, has_protocol_trace


def oracle_path(run: Run, role: str) -> Path:
    return run.oracle_dir / f"{role}.jsonl"


def load_oracle_rows_by_id(path: Path) -> dict[str, QueryResult]:
    return {row.query_id: row for row in read_jsonl(path, QueryResult.from_dict)}


def merge_oracle_rows(
    queries: list[Query],
    cached_oracle_rows: dict[str, QueryResult],
    new_oracle_rows: list[QueryResult],
) -> list[QueryResult]:
    merged = dict(cached_oracle_rows)
    for row in new_oracle_rows:
        merged[row.query_id] = row
    query_ids = {query.query_id for query in queries}
    extra_rows = [row for query_id, row in merged.items() if query_id not in query_ids]
    ordered_rows = [merged[query.query_id] for query in queries if query.query_id in merged]
    extra_rows.sort(key=lambda row: row.query_id)
    return ordered_rows + extra_rows


def stage_oracle(
    run: Run,
    *,
    split: str = "selection_holdout",
    limit: int | None = None,
    backfill: bool = False,
    force: bool = False,
    roles: tuple[str, ...] = ("M_lo", "M_hi"),
) -> None:
    """Stage 4 (Part I M2 pilot on H; Part II on calib/test): construct oracle labels."""
    from llm_routing.oracle import run_oracle_inference

    setting = run.setting()
    protocol = get_protocol(setting)
    pool = setting["pool"]
    queries = select_split(run, split, limit)
    part = "I" if split == "selection_holdout" else "II"
    tag = ""
    if backfill:
        tag += " [backfill]"
    print(f"[{part}/oracle] split={split}  n={len(queries)}{tag}")

    run.oracle_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "split": split,
        "limit": limit,
        "n": len(queries),
        "backfill": backfill,
        "artifact_version": ARTIFACT_VERSION,
        "models": {},
    }

    for role in roles:
        if role not in ("M_lo", "M_hi"):
            raise ValueError(f"role must be M_lo or M_hi, got {role!r}")
        model_id = pool[role]
        path = oracle_path(run, role)
        cached_oracle_rows = load_oracle_rows_by_id(path)
        queries_to_infer = [
            query
            for query in queries
            if force or not (
                (cached_row := cached_oracle_rows.get(query.query_id)) is not None
                and has_protocol_trace(cached_row.model_response)
            )
        ]
        skip_count = len(queries) - len(queries_to_infer)
        print(f"[{part}/oracle] {role} ← {model_id}  run={len(queries_to_infer)}  skip={skip_count}")
        new_oracle_rows = run_oracle_inference(model_id, queries_to_infer, protocol) if queries_to_infer else []
        merged_rows = merge_oracle_rows(queries, cached_oracle_rows, new_oracle_rows)
        write_jsonl(path, merged_rows, QueryResult.to_dict)
        meta["models"][role] = model_id

    write_json(run.oracle_dir / "meta.json", meta)
    run.stage_done(
        "oracle",
        part=part,
        step="4",
        split=split,
        n=len(queries),
        limit=limit,
        backfill=backfill,
    )


def stage_signal_extraction_independent(
    run: Run,
    *,
    limit: int | None = None,
    allow_full_corpus: bool = False,
) -> Path:
    """Stage 5A: model-independent φ(q) on R_c ∪ R_t (holdout excluded)."""
    _, _, partition = load_corpus_artifacts(run.corpus_dir)
    if partition and partition.get("calib") and partition.get("test"):
        query_ids, n_calib, n_test = (
            eval_query_ids(partition)[0],
            len(partition["calib"]),
            len(partition["test"]),
        )
        print(f"[II/5A] split=calib+test  |R_c|={n_calib}  |R_t|={n_test}  holdout excluded")
    elif allow_full_corpus:
        corpus, _, _ = load_corpus_artifacts(run.corpus_dir)
        query_ids = [q.query_id for q in corpus]
        print("[II/5A] WARNING: full corpus (smoke only — not for analysis)")
    else:
        raise ValueError(
            f"M3 eval required — run: python run.py eval --run {run.root}"
        )
    if limit:
        query_ids = query_ids[:limit]

    print(f"[II/5A] n={len(query_ids)}")
    from llm_routing.signals.phi import run_model_independent

    out = run_model_independent(
        run.root,
        query_ids=query_ids,
        allow_full_corpus=allow_full_corpus,
    )
    run.stage_done(
        "model_independent",
        part="II",
        step="5A",
        n=len(query_ids),
        output=str(out.relative_to(run.root)),
    )
    print(f"[II/5A] → {out}")
    return out


def stage_signal_extraction_dependent(
    run: Run,
    *,
    role: str = "M_lo",
    temperature: float = 1.0,
    metrics_version: str | None = None,
) -> Path:
    """Stage 5B: model-dependent ψ(q) from immutable oracle trace."""
    from llm_routing.signals.psi import extract_model_response_signals
    from llm_routing.signals.psi.protocol import METRICS_VERSION

    metrics_version = metrics_version or METRICS_VERSION
    print(f"[II/5B] role={role}  temperature={temperature}  metrics={metrics_version}")
    out = extract_model_response_signals(
        run.root,
        pool_role=role,
        temperature=temperature,
        metrics_version=metrics_version,
    )
    run.stage_done(
        "model_dependent",
        part="II",
        step="5B",
        role=role,
        temperature=temperature,
        metrics_version=metrics_version,
        output=str(out.relative_to(run.root)),
    )
    print(f"[II/5B] → {out}")
    return out


def stage_signal_extraction_cross_model(
    run: Run,
    *,
    metrics_version: str | None = None,
) -> Path:
    """Stage 5C: cross-model χ(q) from joined ψ — offline analysis only."""
    from llm_routing.signals.chi import CROSS_MODEL_METRICS_VERSION, extract_cross_model_signals

    mv = metrics_version or CROSS_MODEL_METRICS_VERSION
    print(f"[II/5C] metrics={mv}")
    out = extract_cross_model_signals(run.root, metrics_version=mv)
    run.stage_done(
        "cross_model",
        part="II",
        step="5C",
        metrics_version=mv,
        output=str(out.relative_to(run.root)),
    )
    print(f"[II/5C] → {out}")
    return out


stage_model_independent = stage_signal_extraction_independent
stage_model_dependent = stage_signal_extraction_dependent
stage_cross_model = stage_signal_extraction_cross_model


def run_development(
    run: Run, *,
    oracle_limit: int | None = None, signal_limit: int | None = None,
    skip_cross_model: bool = False,
) -> Path:
    for split in ("calib", "test"):
        stage_oracle(run, split=split, limit=oracle_limit)
    stage_signal_extraction_independent(run, limit=signal_limit)
    stage_signal_extraction_dependent(run, role="M_lo")
    stage_signal_extraction_dependent(run, role="M_hi")
    if not skip_cross_model:
        stage_signal_extraction_cross_model(run)
    return run.root
