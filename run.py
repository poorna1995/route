#!/usr/bin/env python3
"""CLI: python run.py all --smoke  |  python run.py prepare --run experiments/runs/<id>"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_routing.pipeline import (
    ROOT,
    Run,
    run_all,
    stage_lock_eval,
    stage_oracle,
    stage_prepare,
    stage_scorecard,
    write_selection_report,
)


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
    s.set_defaults(
        func=lambda a: (
            stage_oracle(
                Run.open(a.run),
                split=a.split,
                limit=a.limit if a.limit is not None else (20 if a.smoke else None),
                mock=a.mock,
            ),
            0,
        )[1]
    )

    s = sub.add_parser("select", help="M2: aggregate scorecards → selection_report.json")
    s.set_defaults(func=lambda a: (print(write_selection_report()), 0)[1])

    s = sub.add_parser("lock-eval", help="M3: split C\\H into calib + test (winning run only)")
    s.add_argument("--run", type=Path, required=True)
    s.set_defaults(func=lambda a: (stage_lock_eval(Run.open(a.run)), 0)[1])

    s = sub.add_parser("scorecard")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(
        func=lambda a: (print(json.dumps(stage_scorecard(Run.open(a.run)), indent=2)) if a.json else None, 0)[1]
    )

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
