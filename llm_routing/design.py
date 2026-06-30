"""Experimental Design."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import llm_routing.paths as paths
from llm_routing.corpus import (
    Query,
    QueryResult,
    load_corpus,
    load_corpus_artifacts,
    partition_eval,
    partition_holdout,
    prepare_corpus,
    read_jsonl,
    resolve_test_size,
    save_corpus,
    select_split,
    validate_holdout,
)
from llm_routing.paths import SELECTION_REPORT_PATH
from llm_routing.routing_labels import routing_bucket_name
from llm_routing.run import Run, iter_run_dirs, utc_now, write_json
from llm_routing.setting import (
    corpus_preparation_cfg,
    corpus_spec_from_setting,
    eval_partition_complete,
    freeze_eval_partition,
    freeze_partition_ids,
    frozen_holdout_ids,
    get_gates,
    get_tie_break,
    holdout_size,
    load_defaults,
    partition_cfg_for_m3_lock,
    save_setting,
)


def compute_scorecard(
    queries: list[Query],
    lo: list[QueryResult],
    hi: list[QueryResult],
    gates: dict[str, float],
) -> dict[str, Any]:
    low_model_by_id = {r.query_id: r for r in lo}
    high_model_by_id = {r.query_id: r for r in hi}
    buckets: Counter[str] = Counter()
    for query in queries:
        low_correct = low_model_by_id[query.query_id].correct
        high_correct = high_model_by_id[query.query_id].correct
        buckets[routing_bucket_name(low_correct, high_correct)] += 1
    n = len(queries)
    acc_lo = sum(r.correct for r in lo) / n
    acc_hi = sum(r.correct for r in hi) / n
    acc_gap = acc_hi - acc_lo
    opp = buckets["opportunity"] / n
    too_hard_rate = buckets["too_hard"] / n
    min_gap = gates["min_accuracy_gap"]
    return {
        "n": n,
        "acc_lo": round(acc_lo, 4),
        "acc_hi": round(acc_hi, 4),
        "acc_gap": round(acc_gap, 4),
        "buckets": dict(buckets),
        "opportunity_rate": round(opp, 4),
        "too_hard_rate": round(too_hard_rate, 4),
        "gates": gates,
        "gate_c_pass": acc_gap >= min_gap,
        "gate_d_pass": (
            opp >= gates["opportunity_min"]
            and too_hard_rate < gates["too_hard_max"]
        ),
    }


def message(scorecard: dict[str, Any], setting: dict[str, Any]) -> dict[str, Any]:
    """Gate A–D pass/fail lines for selection report."""
    passed: list[str] = []
    failed: list[str] = []

    if setting.get("protocol", {}).get("grading", {}).get("method") == "mcq":
        passed.append("gate_a: objective_mcq grading")
    else:
        failed.append("gate_a: grading is not mcq")

    passed.append("gate_b: frozen protocol layers (Phase A defaults)")

    gates = scorecard["gates"]
    min_gap = gates["min_accuracy_gap"]
    gap = scorecard["acc_gap"]
    if scorecard["gate_c_pass"]:
        passed.append(
            f"gate_c: acc_gap={gap:.3f} >= {min_gap:.3f} "
            f"(acc_lo={scorecard['acc_lo']:.3f}, acc_hi={scorecard['acc_hi']:.3f})"
        )
    else:
        failed.append(
            f"gate_c: acc_gap={gap:.3f} < {min_gap:.3f} "
            f"(acc_lo={scorecard['acc_lo']:.3f}, acc_hi={scorecard['acc_hi']:.3f})"
        )

    opp, hard = scorecard["opportunity_rate"], scorecard["too_hard_rate"]
    if scorecard["gate_d_pass"]:
        passed.append(
            f"gate_d: opportunity_rate={opp:.3f} >= {gates['opportunity_min']}, "
            f"too_hard_rate={hard:.3f} < {gates['too_hard_max']}"
        )
    else:
        if opp < gates["opportunity_min"]:
            failed.append(
                f"gate_d: opportunity_rate={opp:.3f} < {gates['opportunity_min']}"
            )
        if hard >= gates["too_hard_max"]:
            failed.append(f"gate_d: too_hard_rate={hard:.3f} >= {gates['too_hard_max']}")

    return {"pass": passed, "fail": failed}


def _latest_scorecards(runs_root: Path | None = None) -> dict[str, tuple[str, dict[str, Any]]]:
    root = runs_root or paths.RUNS_ROOT
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for run_dir in iter_run_dirs(root):
        path = run_dir / "scorecard.json"
        if not path.exists():
            continue
        sc = json.loads(path.read_text(encoding="utf-8"))
        dataset = sc.get("dataset")
        run_id = sc.get("run_id", run_dir.name)
        if not dataset:
            continue
        ts = sc.get("timestamp", "")
        prev_ts = latest.get(dataset, ("", {}))[1].get("timestamp", "") if dataset in latest else ""
        if dataset not in latest or ts > prev_ts:
            latest[dataset] = (run_id, sc)
    return latest


def pick_winner(candidates: list[dict[str, Any]], tie_break: dict[str, Any]) -> str | None:
    passers = [c for c in candidates if c["passed"]]
    if not passers:
        return None
    order = tie_break.get("order", ["acc_gap", "opportunity_rate"])
    prefer = tie_break.get("prefer", [])

    def sort_key(entry: dict[str, Any]) -> tuple:
        sc = entry["scorecard"]
        metrics = tuple(-float(sc.get(k, 0)) for k in order)
        pref = prefer.index(entry["candidate"]) if entry["candidate"] in prefer else len(prefer)
        return metrics + (pref,)

    passers.sort(key=sort_key)
    return passers[0]["candidate"]


def build_selection_report(*, runs_root: Path | None = None) -> dict[str, Any]:
    root = runs_root or paths.RUNS_ROOT
    defaults = load_defaults()
    tie_break = get_tie_break(defaults)
    latest = _latest_scorecards(root)

    candidates: list[dict[str, Any]] = []
    for dataset, (run_id, sc) in sorted(latest.items()):
        run = Run.open(root / run_id)
        setting = run.setting()
        msg = message(sc, setting)
        passed = sc["gate_c_pass"] and sc["gate_d_pass"]
        candidates.append(
            {
                "candidate": dataset,
                "run_id": run_id,
                "passed": passed,
                "message": msg,
                "scorecard": {
                    k: sc[k]
                    for k in (
                        "acc_lo", "acc_hi", "acc_gap", "buckets",
                        "opportunity_rate", "too_hard_rate", "gate_c_pass", "gate_d_pass", "n",
                    )
                },
            }
        )

    winner = pick_winner(candidates, tie_break)
    return {
        "timestamp": utc_now(),
        "tie_break": tie_break,
        "candidates": candidates,
        "winner": winner,
    }


def stage_selection_report(*, runs_root: Path | None = None) -> Path:
    report = build_selection_report(runs_root=runs_root)
    write_json(SELECTION_REPORT_PATH, report)
    print(f"[M2/select] winner={report['winner']!r}  → {SELECTION_REPORT_PATH}")
    for c in report["candidates"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {c['candidate']}: {mark}")
    return SELECTION_REPORT_PATH


def stage_prepare(run: Run, *, force_partition: bool = False) -> None:
    """M1: load C and sample selection holdout H only."""
    setting = run.setting()
    spec = corpus_spec_from_setting(setting)
    part_cfg = setting["partition"]

    print(f"[M1/prepare] {spec.name}  |C| loading...")
    corpus_raw = load_corpus(spec)
    prep_cfg = corpus_preparation_cfg(setting)
    corpus, prep_meta = prepare_corpus(
        corpus_raw,
        prep_cfg,
        seed=int(part_cfg.get("seed", 42)),
    )
    n = len(corpus)
    if prep_meta.get("n_dropped_invalid"):
        print(f"[M1/prepare] dropped invalid={prep_meta['n_dropped_invalid']}")
    if prep_meta.get("subsampled"):
        print(
            f"[M1/prepare] subsampled {prep_meta['n_loaded']} → {n} "
            f"(target={prep_meta['subsample_target']}, stratify={prep_cfg.get('stratify_metadata_key')})"
        )
    print(f"[M1/prepare] |C|={n}")

    frozen = None if force_partition else frozen_holdout_ids(setting)
    if frozen:
        validate_holdout(corpus, frozen)
        partition = {"selection_holdout": frozen}
        print(f"[M1/prepare] reuse frozen holdout  |H|={len(frozen)}")
    else:
        sel_n = holdout_size(setting, n)
        print(f"[M1/prepare] holdout only  |H|={sel_n}  (calib/test at M3 on winner)")
        partition = partition_holdout(
            corpus,
            method=part_cfg.get("method", "split_dataset"),
            seed=int(part_cfg.get("seed", 42)),
            selection_n=sel_n,
        )
        save_setting(run.setting_path, freeze_partition_ids(setting, partition))

    save_corpus(
        run.corpus_dir,
        corpus,
        {
            "dataset": spec.name,
            "corpus_size": n,
            "run_id": run.run_id,
            "partition_phase": "m1",
            "corpus_preparation": prep_meta,
        },
        partition,
    )
    run.stage_done("prepare", part="I", step="M1", corpus_size=n, holdout_n=len(partition["selection_holdout"]))


def stage_eval(run: Run) -> None:
    """M3: on winning benchmark, split C \\ H into R_c and R_t."""
    setting = run.setting()
    if eval_partition_complete(setting):
        print("[M3/eval] eval partition already frozen")
        return

    corpus, _, partition = load_corpus_artifacts(run.corpus_dir)
    holdout = partition["selection_holdout"]
    part_cfg = setting["partition"]
    n_rest = len(corpus) - len(holdout)
    test_n = resolve_test_size(partition_cfg_for_m3_lock(setting), eval_pool_n=n_rest)
    print(f"[M3/eval] |C\\H|={n_rest}  R_t={test_n}  R_c={n_rest - test_n}")

    full = partition_eval(
        corpus,
        holdout,
        method=part_cfg.get("method", "split_dataset"),
        seed=int(part_cfg.get("seed", 42)),
        test_n=test_n,
    )
    save_setting(run.setting_path, freeze_eval_partition(setting, full, resolved_test_n=test_n))
    save_corpus(
        run.corpus_dir,
        corpus,
        {"dataset": setting["corpus"]["dataset"], "corpus_size": len(corpus), "run_id": run.run_id, "partition_phase": "m3"},
        full,
    )
    run.stage_done(
        "eval",
        part="I",
        step="M3",
        holdout_n=len(full["selection_holdout"]),
        calib_n=len(full["calib"]),
        test_n=len(full["test"]),
    )


def stage_scorecard(run: Run) -> dict[str, Any]:
    """M2: feasibility scorecard on oracle pilot split."""
    meta = json.loads((run.oracle_dir / "meta.json").read_text(encoding="utf-8"))
    split, limit = meta["split"], meta.get("limit")
    queries = select_split(run, split, limit)
    lo = read_jsonl(run.oracle_dir / "M_lo.jsonl", QueryResult.from_dict)
    hi = read_jsonl(run.oracle_dir / "M_hi.jsonl", QueryResult.from_dict)

    report = compute_scorecard(queries, lo, hi, get_gates(run.setting()))
    report.update(
        run_id=run.run_id,
        split=split,
        dataset=run.setting()["corpus"]["dataset"],
        pool=run.setting()["pool"],
        timestamp=utc_now(),
    )
    write_json(run.scorecard_path, report)
    run.stage_done(
        "scorecard",
        part="I",
        step="M2",
        gate_c_pass=report["gate_c_pass"],
        gate_d_pass=report["gate_d_pass"],
    )

    print(f"\n[M2/scorecard] lo={report['acc_lo']:.1%} hi={report['acc_hi']:.1%} gap={report['acc_gap']:+.1%}")
    print(f"[M2/scorecard] buckets={report['buckets']}  C={report['gate_c_pass']}  D={report['gate_d_pass']}")
    return report
