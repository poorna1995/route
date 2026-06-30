#!/usr/bin/env python3
"""CLI: python run.py all --smoke  |  python run.py prepare --run experiments/runs/<id>"""

from __future__ import annotations

import llm_routing.hf_env  # noqa: F401 — before hub imports

import argparse
import json
from pathlib import Path

from llm_routing.paths import ROOT
from llm_routing.run import (
    Run,
    run_all,
    run_development,
    stage_cross_model,
    stage_eval,
    stage_evaluate,
    stage_model_dependent,
    stage_model_independent,
    stage_model_independent_all,
    stage_oracle,
    stage_prepare,
    stage_scorecard,
    stage_selection_report,
    stage_signal_validation,
)


def _route_demo(args: argparse.Namespace) -> int:
    from llm_routing.deploy.policy import load_policy, resolve_policy_path
    from llm_routing.deploy.router import route_query

    run = Run.open(args.run)
    policy_path = resolve_policy_path(run.root, args.policy)
    policy = load_policy(policy_path)
    decision = route_query(policy, run.root, args.query_id)
    print(
        json.dumps(
            {
                "query_id": decision.query_id,
                "score": decision.score,
                "route_hi": decision.route_hi,
                "model": decision.model,
                "threshold": policy.threshold,
                "policy_path": str(policy_path.relative_to(run.root)),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_setting(s: argparse.ArgumentParser, default: str = "experiments/candidates/arc.yaml") -> None:
        s.add_argument("--setting", type=Path, default=ROOT / default)

    s = sub.add_parser("new", help="Create run dir with full setting snapshot")
    add_setting(s)
    s.add_argument("--name", default="run")
    s.set_defaults(func=lambda a: (print(Run.create(a.setting, name=a.name).root), 0)[1])

    s = sub.add_parser("prepare")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--force-partition", action="store_true")
    s.set_defaults(func=lambda a: (stage_prepare(Run.open(a.run), force_partition=a.force_partition), 0)[1])

    s = sub.add_parser("oracle")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--split", default="selection_holdout", choices=list(("selection_holdout", "calib", "test")))
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--mock", action="store_true", help="fake oracle outputs (no GPU/HF weights)")
    s.add_argument("--backfill", action="store_true", help="only infer rows missing model_response.trace")
    s.add_argument("--force", action="store_true", help="re-infer all queries in split (ignore existing trace)")
    s.add_argument("--role", choices=("M_lo", "M_hi", "both"), default="both")
    s.set_defaults(
        func=lambda a: (
            stage_oracle(
                Run.open(a.run),
                split=a.split,
                limit=a.limit if a.limit is not None else (20 if a.smoke else None),
                mock=a.mock,
                backfill=a.backfill,
                force=a.force,
                roles=("M_lo", "M_hi") if a.role == "both" else (a.role,),
            ),
            0,
        )[1]
    )

    s = sub.add_parser("selection-report", help="M2: aggregate scorecards → selection_report.json")
    s.set_defaults(func=lambda a: (print(stage_selection_report()), 0)[1])

    s = sub.add_parser("eval", help="M3: split C\\H into calib + test (winning run only)")
    s.add_argument("--run", type=Path, required=True)
    s.set_defaults(func=lambda a: (stage_eval(Run.open(a.run)), 0)[1])

    s = sub.add_parser("scorecard")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(
        func=lambda a: (print(json.dumps(stage_scorecard(Run.open(a.run)), indent=2)) if a.json else None, 0)[1]
    )

    s = sub.add_parser(
        "model-independent",
        help="Stage 5A (model-independent / H1): query-derived φ(q) extraction",
    )
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--mock-embed", action="store_true", help="deterministic mock embeddings (no sentence-transformers)")
    s.add_argument(
        "--allow-full-corpus",
        action="store_true",
        help="smoke only: skip M3 split guard (not for analysis)",
    )
    s.set_defaults(
        func=lambda a: (
            stage_model_independent(
                Run.open(a.run),
                mock_embed=True if a.mock_embed or a.smoke else False,
                limit=a.limit if a.limit is not None else (5 if a.smoke else None),
                allow_full_corpus=a.allow_full_corpus or a.smoke,
            ),
            0,
        )[1]
    )

    s = sub.add_parser(
        "model-independent-all",
        help="Stage 5A for all candidates: prepare → eval → φ(q) on R_c ∪ R_t",
    )
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--mock-embed", action="store_true")
    s.add_argument("--force-partition", action="store_true")
    s.set_defaults(
        func=lambda a: (
            stage_model_independent_all(
                mock_embed=True if a.mock_embed or a.smoke else False,
                limit=a.limit if a.limit is not None else (5 if a.smoke else None),
                force_partition=a.force_partition,
                smoke=a.smoke,
            ),
            0,
        )[1]
    )

    s = sub.add_parser(
        "model-dependent",
        help="Stage 5B (model-dependent / H2): ψ metrics from oracle trace (CPU)",
    )
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--role", default="M_lo", choices=("M_lo", "M_hi"))
    s.add_argument("--temperature", type=float, default=1.0)
    s.set_defaults(
        func=lambda a: (
            stage_model_dependent(
                Run.open(a.run),
                role=a.role,
                temperature=a.temperature,
            ),
            0,
        )[1]
    )

    s = sub.add_parser(
        "cross-model",
        help="Stage 5C (cross-model / H3): χ metrics from joined ψ signals (CPU)",
    )
    s.add_argument("--run", type=Path, required=True)
    s.set_defaults(
        func=lambda a: (stage_cross_model(Run.open(a.run)), 0)[1]
    )

    s = sub.add_parser(
        "signal-validation",
        help="Stage 6: validate φ/ψ/χ association with r(q) on R_c",
    )
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--cv-folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(
        func=lambda a: (
            print(
                json.dumps(
                    stage_signal_validation(
                        Run.open(a.run),
                        cv_folds=a.cv_folds,
                        seed=a.seed,
                    ),
                    indent=2,
                )
            ),
            0,
        )[1]
    )

    s = sub.add_parser(
        "evaluate",
        help="Part IV — evaluate frozen router on R_t (accuracy, cost, Pareto)",
    )
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--policy", type=Path, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(
        func=lambda a: (
            stage_evaluate(
                Run.open(a.run),
                policy_path=a.policy,
                limit=a.limit,
            ),
            0,
        )[1]
    )

    s = sub.add_parser(
        "develop",
        help="Part II — run Stages 4→5 (+ optional χ) on an M3-locked run",
    )
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--mock-oracle", action="store_true")
    s.add_argument("--mock-embed", action="store_true")
    s.add_argument("--oracle-limit", type=int, default=None)
    s.add_argument("--signal-limit", type=int, default=None)
    s.add_argument("--skip-cross-model", action="store_true")
    s.set_defaults(
        func=lambda a: (
            print(
                run_development(
                    Run.open(a.run),
                    mock_oracle=a.mock_oracle,
                    mock_embed=a.mock_embed,
                    oracle_limit=a.oracle_limit,
                    signal_limit=a.signal_limit,
                    skip_cross_model=a.skip_cross_model,
                )
            ),
            0,
        )[1]
    )

    s = sub.add_parser(
        "route-demo",
        help="Part III — replay frozen policy for one query (loads signals from run dir)",
    )
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--query-id", type=str, required=True)
    s.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="policy.json path (default: routing/policy.json)",
    )
    s.set_defaults(func=_route_demo)

    s = sub.add_parser("all", help="new + prepare + oracle + scorecard")
    add_setting(s)
    s.add_argument("--name", default="pilot")
    s.add_argument("--split", default="selection_holdout")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--force-partition", action="store_true")
    s.add_argument("--mock", action="store_true", help="fake oracle outputs (local dry-run)")
    s.set_defaults(
        func=lambda a: (
            print(
                run_all(
                    a.setting,
                    name=a.name,
                    smoke=a.smoke,
                    limit=a.limit,
                    split=a.split,
                    force_partition=a.force_partition,
                    mock=a.mock,
                ).root
            ),
            0,
        )[1]
    )

    s = sub.add_parser("resume")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--force-partition", action="store_true")
    s.set_defaults(func=_resume)

    args = p.parse_args()
    return args.func(args)


def _resume(args: argparse.Namespace) -> int:
    run = Run.open(args.run)
    stages = run.manifest().get("stages", {})
    cfg = run.manifest().get("config", {})
    if "prepare" not in stages:
        stage_prepare(run, force_partition=args.force_partition)
    if "oracle" not in stages:
        stage_oracle(
            run,
            split=cfg.get("split", "selection_holdout"),
            limit=cfg.get("limit"),
            mock=cfg.get("mock", False),
        )
    if "scorecard" not in stages:
        stage_scorecard(run)
    print(run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
