"""Stage 6: validate predefined φ/ψ/χ association with oracle routing need r(q)."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from llm_routing.corpus import QueryResult, load_corpus_artifacts, read_jsonl
from llm_routing.routing_labels import routing_bucket_name, routing_oracle_r
from llm_routing.signal_schema import CROSS_COLUMNS, LABEL_COLUMNS, QUERY_BLOCK_COLUMNS, QUERY_COLUMNS
from llm_routing.signals.phi.core import JSONL_BLOCKS, flatten_query_row
from llm_routing.signals.psi.protocol import MODEL_RESPONSE_METRIC_KEYS
from llm_routing.signals.record import load_signals

ANALYSIS_DIR = "signals/analysis"
STAGE6_PURPOSE = (
    "Validate association between predefined unsupervised signals and oracle routing need r(q)."
)

# (family, block_id, columns)
FAMILY_BLOCKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    *(("phi", block, QUERY_BLOCK_COLUMNS[block]) for block in JSONL_BLOCKS),
    ("phi", "all", QUERY_COLUMNS),
    ("psi", "psi_lo", tuple(f"psi.{k}" for k in sorted(MODEL_RESPONSE_METRIC_KEYS))),
    ("psi", "psi_hi", tuple(f"psi_hi.{k}" for k in sorted(MODEL_RESPONSE_METRIC_KEYS))),
    ("chi", "all", CROSS_COLUMNS),
)


def _psi_flat(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in MODEL_RESPONSE_METRIC_KEYS:
        val = metrics.get(key)
        if isinstance(val, bool):
            out[f"{prefix}{key}"] = float(val)
        elif isinstance(val, (int, float)):
            out[f"{prefix}{key}"] = float(val)
    return out


def _chi_flat(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in CROSS_COLUMNS:
        key = col.removeprefix("chi.")
        val = metrics.get(key)
        if isinstance(val, bool):
            out[col] = float(val)
        elif isinstance(val, (int, float)):
            out[col] = float(val)
    return out


def _write_table(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_analysis_tables(run: Any) -> dict[str, Path]:
    """Join Stage 4–5 artifacts into separate calib and test tables."""
    _, _, partition = load_corpus_artifacts(run.corpus_dir)
    if not partition or not partition.get("calib"):
        raise ValueError(f"M3 partition missing — run eval on {run.root}")

    calib_ids = set(partition["calib"])
    test_ids = set(partition.get("test") or [])
    query_ids = sorted(calib_ids | test_ids)

    lo_oracle = {
        row.query_id: row
        for row in read_jsonl(run.oracle_dir / "M_lo.jsonl", QueryResult.from_dict)
    }
    hi_oracle = {
        row.query_id: row
        for row in read_jsonl(run.oracle_dir / "M_hi.jsonl", QueryResult.from_dict)
    }

    phi_path = run.signals_dir / "model_independent.jsonl"
    if not phi_path.exists():
        raise FileNotFoundError(phi_path)
    phi_by_id: dict[str, dict[str, float]] = {}
    for row in read_jsonl(phi_path, lambda r: r):
        phi_by_id[row["query_id"]] = flatten_query_row(row)

    psi_lo_path = run.signals_dir / "model_response_M_lo.jsonl"
    psi_hi_path = run.signals_dir / "model_response_M_hi.jsonl"
    chi_path = run.signals_dir / "cross_model_comparative.jsonl"
    for path in (psi_lo_path, psi_hi_path, chi_path):
        if not path.exists():
            raise FileNotFoundError(path)

    psi_lo = {r.query_id: _psi_flat(r.metrics, "psi.") for r in load_signals(psi_lo_path)}
    psi_hi = {r.query_id: _psi_flat(r.metrics, "psi_hi.") for r in load_signals(psi_hi_path)}
    chi = {r.query_id: _chi_flat(r.metrics) for r in load_signals(chi_path)}

    feature_cols = list(QUERY_COLUMNS) + [
        f"psi.{k}" for k in sorted(MODEL_RESPONSE_METRIC_KEYS)
    ] + [f"psi_hi.{k}" for k in sorted(MODEL_RESPONSE_METRIC_KEYS)] + list(CROSS_COLUMNS)

    out_dir = run.root / ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "split", *LABEL_COLUMNS[1:], *feature_cols]

    calib_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for qid in query_ids:
        lo = lo_oracle.get(qid)
        hi = hi_oracle.get(qid)
        if lo is None or hi is None:
            continue
        split = "calib" if qid in calib_ids else "test"
        y_lo, y_hi = int(lo.correct), int(hi.correct)
        row: dict[str, Any] = {
            "query_id": qid,
            "split": split,
            "r": routing_oracle_r(y_lo, y_hi),
            "bucket": routing_bucket_name(y_lo, y_hi),
            "y_lo": y_lo,
            "y_hi": y_hi,
        }
        for col in feature_cols:
            val = None
            for src in (
                phi_by_id.get(qid, {}),
                psi_lo.get(qid, {}),
                psi_hi.get(qid, {}),
                chi.get(qid, {}),
            ):
                if col in src:
                    val = src[col]
                    break
            row[col] = val
        (calib_rows if split == "calib" else test_rows).append(row)

    calib_path = out_dir / "analysis_table_calib.csv"
    test_path = out_dir / "analysis_table_test.csv"
    _write_table(calib_path, fieldnames, calib_rows)
    _write_table(test_path, fieldnames, test_rows)
    return {"calib": calib_path, "test": test_path}


def _load_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty table: {path}")
    skip = {"query_id", "split", "r", "bucket", "y_lo", "y_hi"}
    features = [c for c in rows[0].keys() if c not in skip]
    return rows, features


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[int]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = _rank(x), _rank([float(v) for v in y])
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[0], qs[1], qs[2]


def _feature_metrics(x: list[float], y: list[int]) -> dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as e:
        raise ImportError("Stage 6 requires scikit-learn (pip install scikit-learn)") from e

    out: dict[str, Any] = {"n": len(y), "n_pos": sum(y), "n_neg": len(y) - sum(y)}
    pos_x = [v for v, r in zip(x, y) if r == 1]
    neg_x = [v for v, r in zip(x, y) if r == 0]
    out["mean_positive"] = statistics.mean(pos_x) if pos_x else None
    out["mean_negative"] = statistics.mean(neg_x) if neg_x else None

    if len(set(y)) < 2:
        out.update(auroc=None, auprc=None, spearman=None, direction=None)
        return out

    out["auroc"] = float(roc_auc_score(y, x))
    out["auprc"] = float(average_precision_score(y, x))
    out["spearman"] = _spearman(x, y)
    if out["mean_positive"] is not None and out["mean_negative"] is not None:
        if out["mean_positive"] > out["mean_negative"]:
            out["direction"] = "positive"
        elif out["mean_positive"] < out["mean_negative"]:
            out["direction"] = "negative"
        else:
            sp = out["spearman"]
            out["direction"] = "positive" if sp and sp > 0 else "negative" if sp and sp < 0 else None
    else:
        out["direction"] = None
    return out


def _column_values(
    rows: list[dict[str, str]],
    col: str,
    *,
    target_col: str = "r",
) -> tuple[list[float], list[int]] | None:
    xs: list[float] = []
    ys: list[int] = []
    for row in rows:
        raw = row.get(col)
        if raw in (None, ""):
            continue
        try:
            xs.append(float(raw))
            ys.append(int(row[target_col]))
        except (TypeError, ValueError):
            continue
    if len(xs) < 5:
        return None
    return xs, ys


def compute_univariates(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    target_col: str = "r",
) -> list[dict[str, Any]]:
    """Level 1: per-feature association with target label."""
    out: list[dict[str, Any]] = []
    for col in features:
        parsed = _column_values(rows, col, target_col=target_col)
        if parsed is None:
            continue
        xs, ys = parsed
        family = col.split(".", 1)[0]
        out.append(
            {
                "feature": col,
                "family": family,
                "target": target_col,
                **_feature_metrics(xs, ys),
            }
        )
    return out


def summarize_families(univariates: list[dict[str, Any]], *, target_col: str = "r") -> list[dict[str, Any]]:
    """Level 2: distribution of per-feature AUROCs within each predefined block."""
    out: list[dict[str, Any]] = []
    for family, block, cols in FAMILY_BLOCKS:
        aurocs = [
            u["auroc"]
            for u in univariates
            if u["feature"] in cols and u.get("auroc") is not None
        ]
        if not aurocs:
            continue
        aurocs_sorted = sorted(aurocs)
        mid = len(aurocs_sorted) // 2
        median = (
            aurocs_sorted[mid]
            if len(aurocs_sorted) % 2
            else (aurocs_sorted[mid - 1] + aurocs_sorted[mid]) / 2
        )
        q25, _, q75 = _quartiles(aurocs_sorted)
        out.append(
            {
                "family": family,
                "block": block,
                "target": target_col,
                "n_features": len(aurocs),
                "mean_auroc": statistics.mean(aurocs),
                "median_auroc": median,
                "std_auroc": statistics.pstdev(aurocs) if len(aurocs) > 1 else 0.0,
                "min_auroc": min(aurocs),
                "max_auroc": max(aurocs),
                "q25_auroc": q25,
                "q75_auroc": q75,
                "iqr_auroc": q75 - q25,
            }
        )
    return out


def _feature_variance(rows: list[dict[str, str]], features: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for col in features:
        values: list[float] = []
        for row in rows:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if len(values) < 2:
            continue
        out.append(
            {
                "feature": col,
                "family": col.split(".", 1)[0],
                "n": len(values),
                "variance": statistics.pvariance(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return out


def validate_signals(table_path: Path) -> dict[str, Any]:
    """Univariate evidence + family summaries on one analysis table (R_c only in Stage 6)."""
    rows, features = _load_table(table_path)
    univariates_by_target: dict[str, list[dict[str, Any]]] = {}
    family_summary_by_target: dict[str, list[dict[str, Any]]] = {}
    for target_col in ("r", "y_lo", "y_hi"):
        target_univariates = compute_univariates(rows, features, target_col=target_col)
        univariates_by_target[target_col] = target_univariates
        family_summary_by_target[target_col] = summarize_families(
            target_univariates, target_col=target_col
        )
    psi_features = [f for f in features if f.startswith("psi.")]
    psi_variance = _feature_variance(rows, psi_features)
    return {
        "meta": {
            "purpose": STAGE6_PURPOSE,
            "split": rows[0].get("split", "calib"),
            "n_queries": len(rows),
            "positive_rate": sum(int(r["r"]) for r in rows) / len(rows),
            "primary_metric": "auroc",
            "source_table": table_path.name,
            "targets_evaluated": ["r", "y_lo", "y_hi"],
        },
        # Backward-compatible aliases for existing readers.
        "univariates": univariates_by_target["r"],
        "family_summary": family_summary_by_target["r"],
        "univariates_by_target": univariates_by_target,
        "family_summary_by_target": family_summary_by_target,
        "psi_variance": psi_variance,
    }


def probe_linear_representations(
    rows: list[dict[str, str]],
    *,
    cv_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Level 3 (secondary): linear readout — joint separability, not primary validation."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError("linear representation probes require scikit-learn") from e

    y = np.array([int(r["r"]) for r in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        return []

    out: list[dict[str, Any]] = []
    n_splits = min(cv_folds, int(y.sum()), int(len(y) - y.sum()))
    if n_splits < 2:
        return []

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for family, block, cols in FAMILY_BLOCKS:
        xs: list[list[float]] = []
        for row in rows:
            vec: list[float] = []
            ok = True
            for col in cols:
                try:
                    vec.append(float(row[col]))
                except (TypeError, ValueError, KeyError):
                    ok = False
                    break
            xs.append(vec if ok else [])

        valid_idx = [i for i, v in enumerate(xs) if v]
        if len(valid_idx) < n_splits * 2:
            continue
        X = np.asarray([xs[i] for i in valid_idx], dtype=float)
        y_sub = y[valid_idx]
        aurocs: list[float] = []
        auprcs: list[float] = []
        for train_i, test_i in kf.split(X, y_sub):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_i])
            X_te = scaler.transform(X[test_i])
            clf = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            )
            clf.fit(X_tr, y_sub[train_i])
            prob = clf.predict_proba(X_te)[:, 1]
            y_te = y_sub[test_i]
            if len(set(y_te.tolist())) < 2:
                continue
            aurocs.append(float(roc_auc_score(y_te, prob)))
            auprcs.append(float(average_precision_score(y_te, prob)))
        if not aurocs:
            continue
        out.append(
            {
                "family": family,
                "block": block,
                "n_features": len(cols),
                "cv_folds": n_splits,
                "cv_auroc_mean": statistics.mean(aurocs),
                "cv_auroc_std": statistics.pstdev(aurocs) if len(aurocs) > 1 else 0.0,
                "cv_auprc_mean": statistics.mean(auprcs),
                "cv_auprc_std": statistics.pstdev(auprcs) if len(auprcs) > 1 else 0.0,
            }
        )
    return out


def stage_signal_validation(
    run: Any,
    *,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage 6: analysis tables + signal informativeness evidence (R_c only)."""
    from llm_routing.run import write_json

    print("[II/6] signal validation  split=calib")
    tables = build_analysis_tables(run)
    calib_path = tables["calib"]
    print(
        f"[II/6] tables → {calib_path.relative_to(run.root)} "
        f"({sum(1 for _ in calib_path.open()) - 1} rows), "
        f"{tables['test'].relative_to(run.root)} "
        f"({sum(1 for _ in tables['test'].open()) - 1} rows)"
    )

    report = validate_signals(calib_path)
    rows, _ = _load_table(calib_path)
    probes = probe_linear_representations(rows, cv_folds=cv_folds, seed=seed)
    report["linear_representation_probes"] = probes

    out_dir = run.root / ANALYSIS_DIR
    write_json(out_dir / "validation_meta.json", report["meta"])
    (out_dir / "univariates.json").write_text(
        json.dumps(report["univariates"], indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "family_summary.json").write_text(
        json.dumps(report["family_summary"], indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "univariates_by_target.json").write_text(
        json.dumps(report["univariates_by_target"], indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "family_summary_by_target.json").write_text(
        json.dumps(report["family_summary_by_target"], indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "psi_variance.json").write_text(
        json.dumps(report["psi_variance"], indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "linear_representation_probes.json").write_text(
        json.dumps(probes, indent=2) + "\n", encoding="utf-8"
    )

    meta = report["meta"]
    print(
        f"[II/6] n={meta['n_queries']}  r_rate={meta['positive_rate']:.3f}  "
        f"features={len(report['univariates'])}  families={len(report['family_summary'])}"
    )
    run.stage_done(
        "signal_validation",
        part="II",
        step="6",
        split="calib",
        n_queries=meta["n_queries"],
        n_features=len(report["univariates"]),
    )
    return report
