"""Post-hoc RH5 analysis — divergence depth and bucket median curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routing.constants import BUCKET_ORDER
from routing.layerwise import depth_fraction_list

# F7 curves omit weak_only for readability.
F7_BUCKETS = tuple(b for b in BUCKET_ORDER if b != "weak_only")


def trace_depth_fraction(rec: dict[str, Any]) -> list[float]:
    """X-axis for F7 — stored in JSONL; derived for legacy traces."""
    if "depth_fraction" in rec:
        return list(rec["depth_fraction"])
    return depth_fraction_list(int(rec["num_layers"]))


def load_traces(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["query_id"])] = rec
    if not out:
        raise ValueError(f"no traces in {path}")
    return out


def bucket_medians(traces: dict[str, dict], merged: pd.DataFrame) -> dict[str, list[float]]:
    n_layers = int(next(iter(traces.values()))["num_layers"])
    medians: dict[str, list[float]] = {}
    for bucket in F7_BUCKETS:
        ids = set(merged.loc[merged["bucket"] == bucket, "query_id"].astype(str))
        rows = [traces[qid] for qid in ids if qid in traces]
        if not rows:
            medians[bucket] = [float("nan")] * n_layers
            continue
        stack = np.array([r["margin"] for r in rows], dtype=float)
        medians[bucket] = np.nanmedian(stack, axis=0).tolist()
    return medians


def divergence_layer(
    medians: dict[str, list[float]],
    *,
    depth_fraction: list[float] | None = None,
    opp_key: str = "opportunity",
    hard_key: str = "too_hard",
) -> dict[str, Any]:
    opp = np.asarray(medians[opp_key], dtype=float)
    hard = np.asarray(medians[hard_key], dtype=float)
    n = len(opp)
    disp = np.abs(opp - hard)
    if n == 0 or not np.isfinite(disp).any():
        return {
            "layer_star": None,
            "fraction_depth": None,
            "n_layers": n,
            "tau": None,
            "pairwise_dispersion": [],
        }
    finite = disp[np.isfinite(disp)]
    tau = float(np.nanpercentile(finite, 95)) if len(finite) else 0.0
    star_idx: int | None = None
    if tau > 0:
        for i, d in enumerate(disp):
            if np.isfinite(d) and d >= tau:
                star_idx = i
                break
    if star_idx is None:
        star_idx = int(np.nanargmax(disp))
    layer_star = star_idx + 1
    if depth_fraction and len(depth_fraction) == n:
        frac = float(depth_fraction[star_idx])
    else:
        frac = layer_star / n if n else None
    return {
        "layer_star": int(layer_star),
        "fraction_depth": round(frac, 4) if frac is not None else None,
        "n_layers": n,
        "tau": tau,
        "pairwise_dispersion": [float(x) if np.isfinite(x) else None for x in disp],
    }


def analyze_formation(
    *,
    trace_path: Path,
    merged_csv: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    traces = load_traces(trace_path)
    merged = pd.read_csv(merged_csv)
    depth_fraction = trace_depth_fraction(next(iter(traces.values())))
    medians = bucket_medians(traces, merged)
    div = divergence_layer(medians, depth_fraction=depth_fraction)

    payload: dict[str, Any] = {
        "analysis": "layerwise_evolution",
        "hypothesis": "RH5",
        "trace_path": str(trace_path),
        "merged_csv": str(merged_csv),
        "depth_fraction": depth_fraction,
        "bucket_medians": medians,
        "divergence": div,
        "interpretation": (
            f"Opportunity vs too-hard median margins diverge near fraction depth "
            f"{div['fraction_depth']} (layer {div['layer_star']}/{div['n_layers']})."
            if div.get("fraction_depth") is not None
            else "No clear divergence layer."
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
