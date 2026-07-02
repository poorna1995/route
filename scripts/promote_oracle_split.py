#!/usr/bin/env python3
"""Split a scratch run's merged oracle into permanent pilot (H) + val (R_c) runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from collections import Counter

from llm_routing.corpus import Query, QueryResult, load_corpus_artifacts, read_jsonl, select_queries, write_jsonl
from llm_routing.routing_labels import routing_bucket_name
from llm_routing.run import utc_now, write_json
from llm_routing.setting import get_gates, load_setting


def compute_scorecard(
    queries: list[Query],
    lo: list[QueryResult],
    hi: list[QueryResult],
    gates: dict[str, float],
) -> dict:
    low_model_by_id = {r.query_id: r for r in lo}
    high_model_by_id = {r.query_id: r for r in hi}
    buckets: Counter[str] = Counter()
    for query in queries:
        buckets[routing_bucket_name(
            low_model_by_id[query.query_id].correct,
            high_model_by_id[query.query_id].correct,
        )] += 1
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


def _oracle_rows(path: Path) -> dict[str, QueryResult]:
    return {r.query_id: r for r in read_jsonl(path, QueryResult.from_dict)}


def _ordered_rows(ids: list[str], by_id: dict[str, QueryResult]) -> list[QueryResult]:
    missing = [qid for qid in ids if qid not in by_id]
    if missing:
        raise ValueError(f"missing {len(missing)} oracle rows, e.g. {missing[0]!r}")
    return [by_id[qid] for qid in ids]


def _write_oracle_bundle(
    dest: Path,
    *,
    split: str,
    ids: list[str],
    lo_by_id: dict[str, QueryResult],
    hi_by_id: dict[str, QueryResult],
    models: dict[str, str],
    mock: bool,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    lo_rows = _ordered_rows(ids, lo_by_id)
    hi_rows = _ordered_rows(ids, hi_by_id)
    write_jsonl(dest / "M_lo.jsonl", lo_rows, QueryResult.to_dict)
    write_jsonl(dest / "M_hi.jsonl", hi_rows, QueryResult.to_dict)
    meta = {
        "split": split,
        "limit": None,
        "n": len(ids),
        "mock": mock,
        "backfill": False,
        "artifact_version": 3,
        "models": models,
    }
    write_json(dest / "meta.json", meta)
    if split == "calib":
        write_jsonl(dest / "M_lo_calib.jsonl", lo_rows, QueryResult.to_dict)
        write_jsonl(dest / "M_hi_calib.jsonl", hi_rows, QueryResult.to_dict)
        write_json(dest / "meta_calib.json", {k: v for k, v in meta.items() if k != "backfill"})


def _scorecard(
    run_id: str,
    split: str,
    queries: list[Query],
    lo: list[QueryResult],
    hi: list[QueryResult],
    setting: dict,
) -> dict:
    report = compute_scorecard(queries, lo, hi, get_gates(setting))
    report.update(
        run_id=run_id,
        split=split,
        dataset=setting["corpus"]["dataset"],
        pool=setting["pool"],
        timestamp=utc_now(),
    )
    return report


def promote(src: Path, *, slug: str, dest_root: Path) -> tuple[Path, Path]:
    src = src.resolve()
    setting = load_setting(src / "setting.yaml")
    corpus, _, partition = load_corpus_artifacts(src / "corpus")
    lo_by_id = _oracle_rows(src / "oracle" / "M_lo.jsonl")
    hi_by_id = _oracle_rows(src / "oracle" / "M_hi.jsonl")
    meta = json.loads((src / "oracle" / "meta.json").read_text(encoding="utf-8"))
    models = meta["models"]
    mock = bool(meta.get("mock", False))

    pilot_id = f"{slug}_oracle_pilot"
    val_id = f"{slug}_oracle_val"
    pilot_root = dest_root / pilot_id
    val_root = dest_root / val_id

    for root in (pilot_root, val_root):
        if root.exists():
            shutil.rmtree(root)
        (root / "corpus").mkdir(parents=True)
        shutil.copy2(src / "corpus/corpus.jsonl", root / "corpus/corpus.jsonl")
        shutil.copy2(src / "corpus/manifest.json", root / "corpus/manifest.json")
        shutil.copy2(src / "corpus/partition.json", root / "corpus/partition.json")
        shutil.copy2(src / "setting.yaml", root / "setting.yaml")

    holdout_ids = partition["selection_holdout"]
    calib_ids = partition["calib"]
    holdout_queries = select_queries(corpus, holdout_ids)
    calib_queries = select_queries(corpus, calib_ids)

    _write_oracle_bundle(
        pilot_root / "oracle",
        split="selection_holdout",
        ids=holdout_ids,
        lo_by_id=lo_by_id,
        hi_by_id=hi_by_id,
        models=models,
        mock=mock,
    )
    _write_oracle_bundle(
        val_root / "oracle",
        split="calib",
        ids=calib_ids,
        lo_by_id=lo_by_id,
        hi_by_id=hi_by_id,
        models=models,
        mock=mock,
    )

    pilot_sc = _scorecard(
        pilot_id, "selection_holdout", holdout_queries,
        _ordered_rows(holdout_ids, lo_by_id),
        _ordered_rows(holdout_ids, hi_by_id),
        setting,
    )
    val_sc = _scorecard(
        val_id, "calib", calib_queries,
        _ordered_rows(calib_ids, lo_by_id),
        _ordered_rows(calib_ids, hi_by_id),
        setting,
    )
    write_json(pilot_root / "scorecard.json", pilot_sc)
    write_json(val_root / "scorecard.json", val_sc)

    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    write_json(
        pilot_root / "manifest.json",
        {
            "run_id": pilot_id,
            "created_at": src_manifest.get("created_at", utc_now()),
            "setting_source": src_manifest.get("setting_source", "experiments/candidates/mmlu_pro.yaml"),
            "promoted_from": src.name,
            "stages": {
                "prepare": src_manifest.get("stages", {}).get("prepare", {}),
                "oracle": {
                    "status": "done",
                    "finished_at": utc_now(),
                    "split": "selection_holdout",
                    "n": len(holdout_ids),
                    "mock": mock,
                },
                "scorecard": {
                    "status": "done",
                    "finished_at": utc_now(),
                    "gate_c_pass": pilot_sc["gate_c_pass"],
                    "gate_d_pass": pilot_sc["gate_d_pass"],
                },
            },
            "config": {"split": "selection_holdout", "mock": mock},
        },
    )
    write_json(
        val_root / "manifest.json",
        {
            "run_id": val_id,
            "created_at": src_manifest.get("created_at", utc_now()),
            "setting_source": src_manifest.get("setting_source", "experiments/candidates/mmlu_pro.yaml"),
            "promoted_from": src.name,
            "stages": {
                "prepare": src_manifest.get("stages", {}).get("prepare", {}),
                "eval": src_manifest.get("stages", {}).get("eval", {}),
                "oracle": {
                    "status": "done",
                    "finished_at": utc_now(),
                    "part": "II",
                    "step": "4",
                    "split": "calib",
                    "n": len(calib_ids),
                    "mock": mock,
                },
                "scorecard": {
                    "status": "done",
                    "finished_at": utc_now(),
                    "gate_c_pass": val_sc["gate_c_pass"],
                    "gate_d_pass": val_sc["gate_d_pass"],
                },
            },
            "config": {"split": "calib", "mock": mock},
        },
    )

    print(f"[pilot] {pilot_root}")
    print(f"  n={pilot_sc['n']}  gap={pilot_sc['acc_gap']:+.1%}  C={pilot_sc['gate_c_pass']}  D={pilot_sc['gate_d_pass']}")
    print(f"[val]   {val_root}")
    print(f"  n={val_sc['n']}  gap={val_sc['acc_gap']:+.1%}  C={val_sc['gate_c_pass']}  D={val_sc['gate_d_pass']}")
    return pilot_root, val_root


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="scratch run with merged oracle jsonl")
    p.add_argument("--slug", default="mmlu_pro", help="dataset slug for permanent name")
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("experiments/runs/permanent/oracle"),
    )
    args = p.parse_args()
    promote(args.run, slug=args.slug, dest_root=args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
