"""Run layout + pipeline stages (prepare → oracle → scorecard → query-derived φ / model-response ψ)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_routing.corpus import (
    Query,
    QueryResult,
    eval_query_ids,
    load_corpus,
    load_corpus_artifacts,
    partition_eval,
    partition_holdout,
    read_jsonl,
    resolve_test_size,
    save_corpus,
    select_queries,
    validate_holdout,
    write_jsonl,
)
from llm_routing.model_response.protocol import ARTIFACT_VERSION, has_protocol_trace
from llm_routing.oracle import run_oracle_inference, run_oracle_inference_mock
from llm_routing.setting import (
    corpus_spec_from_setting,
    eval_partition_complete,
    freeze_eval_partition,
    freeze_partition_ids,
    frozen_holdout_ids,
    get_gates,
    get_protocol,
    get_tie_break,
    holdout_size,
    load_defaults,
    load_setting,
    partition_cfg_for_m3_lock,
    save_setting,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "experiments" / "runs"
PERMANENT_RUNS_ROOT = RUNS_ROOT / "permanent"
SELECTION_REPORT_PATH = RUNS_ROOT / "selection_report.json"
SPLITS = ("selection_holdout", "calib", "test")
SPLIT_SUFFIX = {
    "selection_holdout": "pilot",
    "calib": "val",
    "test": "test",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists()


def iter_run_dirs(runs_root: Path = RUNS_ROOT) -> list[Path]:
    """Scratch timestamp runs plus grouped permanent runs."""
    if not runs_root.exists():
        return []
    found: list[Path] = []
    permanent = runs_root / "permanent"
    if permanent.is_dir():
        for group in sorted(permanent.iterdir()):
            if group.is_dir():
                found.extend(sorted(p for p in group.iterdir() if _is_run_dir(p)))
    found.extend(
        sorted(
            p
            for p in runs_root.iterdir()
            if p.name != "permanent" and not p.name.startswith(".") and _is_run_dir(p)
        )
    )
    return found


def permanent_run_name(dataset: str, stage: str, split: str) -> str:
    """Stable run dir name, e.g. arc_oracle_pilot or mmlu_query_derived_eval."""
    slug = dataset.lower().replace("-", "_").removesuffix("_challenge")
    if stage == "query_derived" and split == "eval":
        return f"{slug}_query_derived_eval"
    suffix = SPLIT_SUFFIX.get(split, split)
    return f"{slug}_{stage}_{suffix}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# --- scorecard ---


def routing_bucket_name(low_model_correct: int, high_model_correct: int) -> str:
    if low_model_correct and high_model_correct:
        return "easy"
    if not low_model_correct and high_model_correct:
        return "opportunity"
    if low_model_correct and not high_model_correct:
        return "lo_only"
    return "too_hard"


def routing_oracle_r(low_model_correct: int, high_model_correct: int) -> int:
    """r(q)=1 iff M_lo failed and M_hi succeeded (opportunity bucket)."""
    return int(low_model_correct == 0 and high_model_correct == 1)


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


# --- M2 selection report ---


def gate_reasons(scorecard: dict[str, Any], setting: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    """Return (reason_pass, reason_fail, passed) for gates A–D."""
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

    ok = scorecard["gate_c_pass"] and scorecard["gate_d_pass"]
    return passed, failed, ok


def _latest_scorecards(runs_root: Path = RUNS_ROOT) -> dict[str, tuple[str, dict[str, Any]]]:
    """dataset → (run_id, scorecard) — latest run per candidate."""
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for run_dir in iter_run_dirs(runs_root):
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


def build_selection_report(*, runs_root: Path = RUNS_ROOT) -> dict[str, Any]:
    """Aggregate M2 scorecards into paper-citeable selection_report.json."""
    defaults = load_defaults()
    tie_break = get_tie_break(defaults)
    latest = _latest_scorecards(runs_root)

    candidates: list[dict[str, Any]] = []
    for dataset, (run_id, sc) in sorted(latest.items()):
        run = Run.open(runs_root / run_id)
        setting = run.setting()
        reason_pass, reason_fail, passed = gate_reasons(sc, setting)
        candidates.append(
            {
                "candidate": dataset,
                "run_id": run_id,
                "passed": passed,
                "reason_pass": reason_pass,
                "reason_fail": reason_fail,
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
        "timestamp": _utc_now(),
        "tie_break": tie_break,
        "candidates": candidates,
        "winner": winner,
    }


def write_selection_report(*, runs_root: Path = RUNS_ROOT) -> Path:
    report = build_selection_report(runs_root=runs_root)
    _write_json(SELECTION_REPORT_PATH, report)
    print(f"[select] winner={report['winner']!r}  → {SELECTION_REPORT_PATH}")
    for c in report["candidates"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {c['candidate']}: {mark}")
    return SELECTION_REPORT_PATH


# --- run context ---


@dataclass(frozen=True)
class Run:
    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def setting_path(self) -> Path:
        return self.root / "setting.yaml"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def oracle_dir(self) -> Path:
        return self.root / "oracle"

    @property
    def signals_dir(self) -> Path:
        return self.root / "signals"

    @property
    def scorecard_path(self) -> Path:
        return self.root / "scorecard.json"

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def setting(self) -> dict[str, Any]:
        return load_setting(self.setting_path)

    def stage_done(self, name: str, **info: Any) -> None:
        m = self.manifest()
        m.setdefault("stages", {})[name] = {"status": "done", "finished_at": _utc_now(), **info}
        _write_json(self.root / "manifest.json", m)

    @classmethod
    def create(
        cls,
        setting_src: Path | str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Run:
        setting_src = Path(setting_src)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rid = f"{stamp}-{name or setting_src.stem}"
        root = RUNS_ROOT / rid
        if root.exists():
            raise FileExistsError(f"run already exists: {root}")
        root.mkdir(parents=True)
        save_setting(root / "setting.yaml", load_setting(setting_src))
        _write_json(
            root / "manifest.json",
            {
                "run_id": rid,
                "created_at": _utc_now(),
                "setting_source": str(
                    setting_src.relative_to(ROOT) if setting_src.is_relative_to(ROOT) else setting_src
                ),
                "stages": {},
                "config": config or {},
            },
        )
        return cls(root)

    @classmethod
    def open(cls, path: Path | str) -> Run:
        root = Path(path)
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"not a run directory: {root}")
        return cls(root)


# --- stages ---


def _select_split(run: Run, split: str, limit: int | None) -> list[Query]:
    corpus, _, partition = load_corpus_artifacts(run.corpus_dir)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split != "selection_holdout" and not (partition.get("calib") and partition.get("test")):
        raise ValueError(
            f"split={split!r} requires M3 eval partition — run: python run.py lock-eval --run {run.root}"
        )
    queries = select_queries(corpus, partition[split])
    return queries[:limit] if limit else queries


def stage_prepare(run: Run, *, force_partition: bool = False) -> None:
    """M1: load C and sample selection holdout H only."""
    setting = run.setting()
    spec = corpus_spec_from_setting(setting)
    part_cfg = setting["partition"]

    print(f"[prepare] {spec.name}  |C| loading...")
    corpus = load_corpus(spec)
    n = len(corpus)
    print(f"[prepare] |C|={n}")

    frozen = None if force_partition else frozen_holdout_ids(setting)
    if frozen:
        validate_holdout(corpus, frozen)
        partition = {"selection_holdout": frozen}
        print(f"[prepare] reuse frozen holdout  |H|={len(frozen)}")
    else:
        sel_n = holdout_size(setting, n)
        print(f"[prepare] M1 holdout only  |H|={sel_n}  (calib/test at M3 on winner)")
        partition = partition_holdout(
            corpus,
            method=part_cfg.get("method", "random_split"),
            seed=int(part_cfg.get("seed", 42)),
            selection_n=sel_n,
        )
        save_setting(run.setting_path, freeze_partition_ids(setting, partition))

    save_corpus(
        run.corpus_dir,
        corpus,
        {"dataset": spec.name, "corpus_size": n, "run_id": run.run_id, "partition_phase": "m1"},
        partition,
    )
    run.stage_done("prepare", corpus_size=n, holdout_n=len(partition["selection_holdout"]))


def stage_lock_eval(run: Run) -> None:
    """M3: on winning benchmark, split C \\ H into R_c and R_t."""
    setting = run.setting()
    if eval_partition_complete(setting):
        print("[lock-eval] eval partition already frozen")
        return

    corpus, _, partition = load_corpus_artifacts(run.corpus_dir)
    holdout = partition["selection_holdout"]
    part_cfg = setting["partition"]
    n_rest = len(corpus) - len(holdout)
    test_n = resolve_test_size(partition_cfg_for_m3_lock(setting), eval_pool_n=n_rest)
    print(f"[lock-eval] |C\\H|={n_rest}  R_t={test_n}  R_c={n_rest - test_n}")

    full = partition_eval(
        corpus,
        holdout,
        method=part_cfg.get("method", "random_split"),
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
        "lock_eval",
        holdout_n=len(full["selection_holdout"]),
        calib_n=len(full["calib"]),
        test_n=len(full["test"]),
    )


def _oracle_path(run: Run, role: str) -> Path:
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
    mock: bool = False,
    backfill: bool = False,
    force: bool = False,
    roles: tuple[str, ...] = ("M_lo", "M_hi"),
) -> None:
    setting = run.setting()
    protocol = get_protocol(setting)
    pool = setting["pool"]
    queries = _select_split(run, split, limit)
    run_oracle = run_oracle_inference_mock if mock else run_oracle_inference
    tag = " [mock]" if mock else ""
    if backfill:
        tag += " [backfill]"
    print(f"[oracle] split={split}  n={len(queries)}{tag}")

    run.oracle_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "split": split,
        "limit": limit,
        "n": len(queries),
        "mock": mock,
        "backfill": backfill,
        "artifact_version": ARTIFACT_VERSION,
        "models": {},
    }

    for role in roles:
        if role not in ("M_lo", "M_hi"):
            raise ValueError(f"role must be M_lo or M_hi, got {role!r}")
        model_id = pool[role]
        path = _oracle_path(run, role)
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
        print(f"[oracle] {role} ← {model_id}  run={len(queries_to_infer)}  skip={skip_count}")
        new_oracle_rows = run_oracle(model_id, queries_to_infer, protocol) if queries_to_infer else []
        merged_rows = merge_oracle_rows(queries, cached_oracle_rows, new_oracle_rows)
        write_jsonl(path, merged_rows, QueryResult.to_dict)
        meta["models"][role] = model_id

    _write_json(run.oracle_dir / "meta.json", meta)
    run.stage_done(
        "oracle",
        split=split,
        n=len(queries),
        limit=limit,
        mock=mock,
        backfill=backfill,
    )


def stage_scorecard(run: Run) -> dict[str, Any]:
    meta = json.loads((run.oracle_dir / "meta.json").read_text(encoding="utf-8"))
    split, limit = meta["split"], meta.get("limit")
    queries = _select_split(run, split, limit)
    lo = read_jsonl(run.oracle_dir / "M_lo.jsonl", QueryResult.from_dict)
    hi = read_jsonl(run.oracle_dir / "M_hi.jsonl", QueryResult.from_dict)

    report = compute_scorecard(queries, lo, hi, get_gates(run.setting()))
    report.update(
        run_id=run.run_id,
        split=split,
        dataset=run.setting()["corpus"]["dataset"],
        pool=run.setting()["pool"],
        timestamp=_utc_now(),
    )
    _write_json(run.scorecard_path, report)
    run.stage_done("scorecard", gate_c_pass=report["gate_c_pass"], gate_d_pass=report["gate_d_pass"])

    print(f"\n[scorecard] lo={report['acc_lo']:.1%} hi={report['acc_hi']:.1%} gap={report['acc_gap']:+.1%}")
    print(f"[scorecard] buckets={report['buckets']}  C={report['gate_c_pass']}  D={report['gate_d_pass']}")
    return report


def stage_query_derived(
    run: Run,
    *,
    mock_embed: bool = False,
    limit: int | None = None,
    allow_full_corpus: bool = False,
) -> Path:
    """Stage 5 (model-independent / H1): query-derived φ(q) on R_c ∪ R_t only (holdout excluded)."""
    _, _, partition = load_corpus_artifacts(run.corpus_dir)
    if partition and partition.get("calib") and partition.get("test"):
        query_ids, n_calib, n_test = (
            eval_query_ids(partition)[0],
            len(partition["calib"]),
            len(partition["test"]),
        )
        print(f"[query-derived] split=calib+test  |R_c|={n_calib}  |R_t|={n_test}  holdout excluded")
    elif allow_full_corpus:
        corpus, _, _ = load_corpus_artifacts(run.corpus_dir)
        query_ids = [q.query_id for q in corpus]
        print("[query-derived] WARNING: full corpus (smoke only — not for analysis)")
    else:
        raise ValueError(
            f"M3 lock-eval required before query-derived — run: python run.py lock-eval --run {run.root}"
        )
    if limit:
        query_ids = query_ids[:limit]

    print(f"[query-derived] n={len(query_ids)}  mock_embed={mock_embed}")
    from llm_routing.query_derived import run_query_derived

    out = run_query_derived(
        run.root,
        query_ids=query_ids,
        mock_embed=mock_embed,
        allow_full_corpus=allow_full_corpus,
    )
    run.stage_done(
        "query_derived",
        n=len(query_ids),
        mock_embed=mock_embed,
        output=str(out.relative_to(run.root)),
    )
    print(f"[query-derived] → {out}")
    return out


def stage_model_response(
    run: Run,
    *,
    role: str = "M_lo",
    temperature: float = 1.0,
    metrics_version: str | None = None,
) -> Path:
    """Stage 5B (model-dependent / H2): ψ metric views from immutable oracle trace (CPU)."""
    from llm_routing.model_response import extract_model_response_signals
    from llm_routing.model_response.protocol import METRICS_VERSION

    metrics_version = metrics_version or METRICS_VERSION
    print(f"[model-response] role={role}  temperature={temperature}  metrics={metrics_version}")
    out = extract_model_response_signals(
        run.root,
        pool_role=role,
        temperature=temperature,
        metrics_version=metrics_version,
    )
    run.stage_done(
        "model_response",
        role=role,
        temperature=temperature,
        metrics_version=metrics_version,
        output=str(out.relative_to(run.root)),
    )
    print(f"[model-response] → {out}")
    return out


def stage_cross_model(
    run: Run,
    *,
    metrics_version: str | None = None,
) -> Path:
    """Stage 5C (cross-model / H3): χ(q) from joined ψ signals — CPU only."""
    from llm_routing.cross_model import CROSS_MODEL_METRICS_VERSION, extract_cross_model_signals

    mv = metrics_version or CROSS_MODEL_METRICS_VERSION
    print(f"[cross-model] metrics={mv}")
    out = extract_cross_model_signals(run.root, metrics_version=mv)
    run.stage_done("cross_model", metrics_version=mv, output=str(out.relative_to(run.root)))
    print(f"[cross-model] → {out}")
    return out


CANDIDATES_DIR = ROOT / "experiments" / "candidates"
QUERY_DERIVED_INDEX = RUNS_ROOT / "query_derived_index.json"


def list_candidate_settings() -> list[Path]:
    return sorted(CANDIDATES_DIR.glob("*.yaml"))


def stage_query_derived_all(
    *,
    mock_embed: bool = False,
    limit: int | None = None,
    force_partition: bool = False,
    smoke: bool = False,
) -> Path:
    """Prepare + lock-eval + query-derived for every benchmark candidate."""
    if smoke and limit is None:
        limit = 5
    index_runs: list[dict[str, Any]] = []

    for setting_path in list_candidate_settings():
        setting = load_setting(setting_path)
        dataset = setting["corpus"]["dataset"]
        tag = setting_path.stem
        run = Run.create(setting_path, name=f"phi-{tag}")
        print(f"\n=== {dataset} ({run.run_id}) ===")
        stage_prepare(run, force_partition=force_partition)
        stage_lock_eval(run)
        out = stage_query_derived(
            run,
            mock_embed=mock_embed,
            limit=limit,
            allow_full_corpus=False,
        )
        meta = json.loads((run.signals_dir / "query_derived_meta.json").read_text(encoding="utf-8"))
        index_runs.append(
            {
                "dataset": dataset,
                "run_id": run.run_id,
                "run_root": str(run.root.relative_to(ROOT)),
                "output": str(out.relative_to(ROOT)),
                "n_queries": meta["n_queries"],
                "n_calib": meta.get("n_calib"),
                "n_test": meta.get("n_test"),
                "holdout_excluded": meta.get("holdout_excluded", True),
            }
        )

    report = {"timestamp": _utc_now(), "mock_embed": mock_embed, "limit": limit, "runs": index_runs}
    _write_json(QUERY_DERIVED_INDEX, report)
    print(f"\n[query-derived-all] {len(index_runs)} datasets → {QUERY_DERIVED_INDEX}")
    return QUERY_DERIVED_INDEX


def run_all(
    setting_src: Path,
    *,
    name: str = "pilot",
    smoke: bool = False,
    limit: int | None = None,
    split: str = "selection_holdout",
    force_partition: bool = False,
    mock: bool = False,
) -> Run:
    cap = limit if limit is not None else (20 if smoke else None)
    run = Run.create(
        setting_src,
        name=name,
        config={"split": split, "limit": cap, "smoke": smoke, "mock": mock},
    )
    print(f"Run → {run.root}")
    stage_prepare(run, force_partition=force_partition)
    stage_oracle(run, split=split, limit=cap, mock=mock)
    stage_scorecard(run)
    return run
