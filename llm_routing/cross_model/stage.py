"""Stage 5C: χ(q) from joined ψ signals — CPU only, no inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_routing.corpus import QueryResult, read_jsonl
from llm_routing.signals import SIGNAL_TYPE_CROSS_MODEL, SignalRecord, load_signals, save_signals

CROSS_MODEL_METRICS_VERSION = "v1"

# Update when adding χ metrics — signals.py legacy flat-row loader can import this set.
CROSS_MODEL_METRIC_KEYS = frozenset({
    "delta_entropy",
    "delta_margin",
    "delta_msp",
    "delta_mean_logprob",
    "prediction_disagreement",
})

CROSS_MODEL_CATEGORICAL_METRIC_KEYS = frozenset({"prediction_disagreement"})


def extract_cross_model_signals(
    run_root: Path,
    *,
    metrics_version: str = CROSS_MODEL_METRICS_VERSION,
) -> Path:
    """Join M_lo and M_hi ψ rows; write signals/cross_model_comparative.jsonl."""
    run_root = Path(run_root)
    signals_dir = run_root / "signals"
    lo_path = signals_dir / "model_response_M_lo.jsonl"
    hi_path = signals_dir / "model_response_M_hi.jsonl"
    for path in (lo_path, hi_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python run.py model-response --run {run_root} --role M_lo "
                "and --role M_hi"
            )

    lo_signals = {row.query_id: row for row in load_signals(lo_path)}
    hi_signals = {row.query_id: row for row in load_signals(hi_path)}
    shared_ids = sorted(set(lo_signals) & set(hi_signals))
    if not shared_ids:
        raise ValueError("no shared query_id between M_lo and M_hi model_response signals")

    lo_oracle = {row.query_id: row for row in read_jsonl(run_root / "oracle" / "M_lo.jsonl", QueryResult.from_dict)}
    hi_oracle = {row.query_id: row for row in read_jsonl(run_root / "oracle" / "M_hi.jsonl", QueryResult.from_dict)}
    missing_lo = [qid for qid in shared_ids if qid not in lo_oracle]
    missing_hi = [qid for qid in shared_ids if qid not in hi_oracle]
    if missing_lo or missing_hi:
        raise ValueError(
            "oracle rows missing for shared query_ids: "
            f"M_lo={missing_lo[:3]} M_hi={missing_hi[:3]}"
        )

    records: list[SignalRecord] = []
    for query_id in shared_ids:
        lo_sig, hi_sig = lo_signals[query_id], hi_signals[query_id]
        lo_m, hi_m = lo_sig.metrics, hi_sig.metrics
        lo_or, hi_or = lo_oracle[query_id], hi_oracle[query_id]
        metrics: dict[str, float | bool] = {}
        for key in ("entropy", "margin", "msp", "mean_logprob"):
            if key in lo_m and key in hi_m:
                metrics[f"delta_{key}"] = float(hi_m[key]) - float(lo_m[key])
        parsed_disagree = (
            lo_or.parsed_answer is not None
            and hi_or.parsed_answer is not None
            and lo_or.parsed_answer != hi_or.parsed_answer
        )
        metrics["prediction_disagreement"] = parsed_disagree
        records.append(
            SignalRecord(
                query_id=query_id,
                signal_type=SIGNAL_TYPE_CROSS_MODEL,
                metrics=metrics,
                metrics_version=metrics_version,
                extractor_version=lo_sig.extractor_version,
            )
        )

    output_path = signals_dir / "cross_model_comparative.jsonl"
    save_signals(output_path, records)
    meta: dict[str, Any] = {
        "signal_type": SIGNAL_TYPE_CROSS_MODEL,
        "metrics_version": metrics_version,
        "n_queries": len(records),
        "sources": {
            "M_lo": str(lo_path.relative_to(run_root)),
            "M_hi": str(hi_path.relative_to(run_root)),
        },
        "metric_keys": sorted(records[0].metrics.keys()) if records else [],
        "categorical_metric_keys": sorted(CROSS_MODEL_CATEGORICAL_METRIC_KEYS),
    }
    (signals_dir / "cross_model_comparative_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
