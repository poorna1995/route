"""Stage 5B: derive ψ metric views from immutable oracle traces (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_routing.corpus import QueryResult, load_corpus_artifacts, read_jsonl
from llm_routing.model_response.protocol import (
    METRICS_VERSION,
    get_protocol_extractor,
    has_protocol_trace,
    model_response_prediction,
    model_response_raw,
)
from llm_routing.signals import SIGNAL_TYPE_MODEL_RESPONSE, SignalRecord, save_signals


def extract_model_response_signals(
    run_root: Path,
    *,
    pool_role: str = "M_lo",
    temperature: float = 1.0,
    metrics_version: str = METRICS_VERSION,
) -> Path:
    run_root = Path(run_root)
    oracle_path = run_root / "oracle" / f"{pool_role}.jsonl"
    if not oracle_path.exists():
        raise FileNotFoundError(oracle_path)

    oracle_rows = read_jsonl(oracle_path, QueryResult.from_dict)
    queries_by_id: dict[str, Any] = {}
    corpus_dir = run_root / "corpus"
    if corpus_dir.exists():
        corpus, _, _ = load_corpus_artifacts(corpus_dir)
        queries_by_id = {q.query_id: q for q in corpus}
    missing_trace = [
        row.query_id for row in oracle_rows if not has_protocol_trace(row.model_response)
    ]
    if missing_trace:
        raise ValueError(
            f"{len(missing_trace)} rows missing model_response.trace in {oracle_path} "
            f"(e.g. {missing_trace[0]}). Re-run oracle with --backfill."
        )

    signal_records: list[SignalRecord] = []
    extractor_name: str | None = None
    protocol_version: str | None = None
    for oracle_row in oracle_rows:
        assert oracle_row.model_response is not None
        protocol_artifact = oracle_row.model_response
        protocol_version = protocol_artifact["protocol_version"]
        extractor = get_protocol_extractor(protocol_version)
        extractor_name = protocol_artifact["extractor"]
        extractor_version = protocol_artifact.get("extractor_version") or protocol_artifact[
            "trace"
        ].get("extractor_version")
        extractor.validate(protocol_artifact["trace"])
        trace = protocol_artifact["trace"]
        metric_values = extractor.compute_metrics(trace, temperature=temperature)
        query = queries_by_id.get(oracle_row.query_id)
        raw = model_response_raw(query) if query is not None else None
        signal_records.append(
            SignalRecord(
                query_id=oracle_row.query_id,
                signal_type=SIGNAL_TYPE_MODEL_RESPONSE,
                metrics=metric_values,
                prediction=model_response_prediction(trace, metric_values),
                raw=raw,
                metrics_version=metrics_version,
                extractor_version=extractor_version,
            )
        )

    signals_dir = run_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    output_path = signals_dir / f"model_response_{pool_role}.jsonl"
    save_signals(output_path, signal_records)

    metric_keys = sorted(signal_records[0].metrics.keys()) if signal_records else []
    meta: dict[str, Any] = {
        "signal_type": SIGNAL_TYPE_MODEL_RESPONSE,
        "metrics_version": metrics_version,
        "protocol_version": protocol_version,
        "extractor": extractor_name,
        "extractor_version": signal_records[0].extractor_version if signal_records else None,
        "pool_role": pool_role,
        "source": str(oracle_path.relative_to(run_root)),
        "n_queries": len(signal_records),
        "temperature": temperature,
        "metric_keys": metric_keys,
        "prediction": {"parsed_answer": "str", "confidence": "float"},
        "raw": {"query": "str", "answer": "str"},
    }
    (signals_dir / f"model_response_{pool_role}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
