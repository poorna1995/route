""" Evaluate frozen router on held-out test R_t."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_routing.corpus import QueryResult, read_jsonl, select_split
from llm_routing.run import Run, utc_now, write_json
from llm_routing.deploy.extract import extract_from_run
from llm_routing.deploy.policy import load_policy, resolve_policy_path
from llm_routing.deploy.router import route_features


def _resolve_policy_path(run: Run, policy_path: Path | str | None) -> Path:
    return resolve_policy_path(run.root, policy_path)


def _kappa_hi(setting: dict[str, Any]) -> float:
    pool = setting.get("pool") or {}
    kappa = pool.get("kappa")
    if kappa is not None:
        try:
            return float(kappa)
        except (TypeError, ValueError):
            pass
    return 1.0


def _oracle_correctness(
    route_hi: bool,
    y_lo: int,
    y_hi: int,
) -> int:
    return y_hi if route_hi else y_lo


def _oracle_star_correctness(y_lo: int, y_hi: int) -> int:
    if y_lo:
        return 1
    if y_hi:
        return 1
    return 0


def evaluate_on_split(
    run: Run,
    *,
    split: str = "test",
    policy_path: Path | str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Grade frozen π on a split; default Part IV = R_t."""
    policy = load_policy(_resolve_policy_path(run, policy_path))
    setting = run.setting()
    kappa = _kappa_hi(setting)

    queries = select_split(run, split, limit)
    lo_by_id = {r.query_id: r for r in read_jsonl(run.oracle_dir / "M_lo.jsonl", QueryResult.from_dict)}
    hi_by_id = {r.query_id: r for r in read_jsonl(run.oracle_dir / "M_hi.jsonl", QueryResult.from_dict)}

    include_query = any(c.startswith("phi.") for c in policy.feature_columns)
    n = 0
    policy_correct = 0
    policy_cost = 0.0
    always_lo_correct = 0
    always_hi_correct = 0
    oracle_correct = 0
    oracle_cost = 0.0
    skipped: list[str] = []

    for query in queries:
        qid = query.query_id
        if qid not in lo_by_id or qid not in hi_by_id:
            skipped.append(qid)
            continue
        y_lo = int(lo_by_id[qid].correct)
        y_hi = int(hi_by_id[qid].correct)
        try:
            features = extract_from_run(
                run.root,
                qid,
                include_query=include_query,
            )
            decision = route_features(policy, qid, features)
        except (FileNotFoundError, KeyError) as exc:
            skipped.append(f"{qid}:{exc}")
            continue

        n += 1
        pc = _oracle_correctness(decision.route_hi, y_lo, y_hi)
        policy_correct += pc
        policy_cost += kappa if decision.route_hi else 0.0

        always_lo_correct += y_lo
        always_hi_correct += y_hi

        oc = _oracle_star_correctness(y_lo, y_hi)
        oracle_correct += oc
        if y_lo:
            oracle_cost += 0.0
        elif y_hi:
            oracle_cost += kappa
        else:
            oracle_cost += 0.0

    def _rate(x: int) -> float:
        return round(x / n, 4) if n else float("nan")

    def _avg_cost(c: float) -> float:
        return round(c / n, 4) if n else float("nan")

    report = {
        "part": "IV",
        "split": split,
        "timestamp": utc_now(),
        "n": n,
        "n_skipped": len(skipped),
        "policy_path": str(_resolve_policy_path(run, policy_path).relative_to(run.root)),
        "kappa_hi": kappa,
        "accuracy": {
            "policy_pi": _rate(policy_correct),
            "always_M_lo": _rate(always_lo_correct),
            "always_M_hi": _rate(always_hi_correct),
            "oracle_pi_star": _rate(oracle_correct),
        },
        "cost": {
            "policy_pi": _avg_cost(policy_cost),
            "always_M_lo": 0.0,
            "always_M_hi": kappa,
            "oracle_pi_star": _avg_cost(oracle_cost),
        },
        "pareto": [
            {"name": "always_M_lo", "accuracy": _rate(always_lo_correct), "cost": 0.0},
            {"name": "always_M_hi", "accuracy": _rate(always_hi_correct), "cost": kappa},
            {"name": "policy_pi", "accuracy": _rate(policy_correct), "cost": _avg_cost(policy_cost)},
            {"name": "oracle_pi_star", "accuracy": _rate(oracle_correct), "cost": _avg_cost(oracle_cost)},
        ],
        "skipped": skipped[:20],
    }
    return report


def stage_evaluate(
    run: Run,
    *,
    policy_path: Path | str | None = None,
    limit: int | None = None,
) -> Path:
    """Part IV: evaluate frozen router on R_t — accuracy, cost, Pareto."""
    report = evaluate_on_split(run, split="test", policy_path=policy_path, limit=limit)
    out = run.root / "evaluation" / "test_report.json"
    write_json(out, report)
    run.stage_done(
        "evaluate",
        part="IV",
        n=report["n"],
        policy_accuracy=report["accuracy"]["policy_pi"],
        output=str(out.relative_to(run.root)),
    )
    print(
        f"[IV/evaluate] n={report['n']}  "
        f"acc_π={report['accuracy']['policy_pi']}  "
        f"cost_π={report['cost']['policy_pi']}  "
        f"→ {out}"
    )
    return out
