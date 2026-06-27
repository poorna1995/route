#!/usr/bin/env python3
"""Unified CLI for the LLM routing research pipeline.

Subcommands:
  oracle          Offline routing-opportunity assessment (C2)
  probes          Prefill signal extraction (C3)
  features        Query complexity features (D46 screening input)
  screen          D46 complexity feature selection on CALIB
  merge           Merge oracle + probes → routing relevance JSON
  doctor          Pre-flight checks on oracle / probe / merge artifacts
  complementarity Cross-family signal ladder analysis
  plot            Paper figures (distributions | roc | scatter | formation | ...)
  interpret       Q2–Q3 interpretation bundle (landscape, overlap, decomposition)
  route-preview   Median-threshold routing sanity check (D37)
  route-eval      Study IV hold-out routing evaluation (EXP-03 / RH4)
  compare-generalization  RH7 dimension transfer across merged regimes
  analyze-formation     RH5 layerwise divergence from JSONL traces
  layerwise-parity      Pre-TEST: lm_head(last_hidden_state) vs out.logits smoke (Llama)
  summarize-c2    Summarize C2 oracle JSON for dataset screening
  verify-logprobs Feasibility check for full-vocabulary logits

Usage:
    python scripts/run.py merge --weak-csv ... --strong-csv ... --oracle ... --output ...
    python scripts/run.py plot distributions --merged-csv ... --output ...
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

# Ensure scripts/ is on path when invoked as `python scripts/run.py`.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing.evaluation import (
    analyze_routing_relevance,
    build_relevance_summary,
    run_calibration_stability,
    run_complementarity_analysis,
    run_failure_analysis,
    run_failure_analysis_bundle,
    run_interpretation_analysis,
    summarize_c2,
)
from routing.model_dependent import run_probe_extraction
from routing.layerwise import run_layerwise_extraction, run_terminal_parity_smoke
from routing.model_independent import run_feature_extraction, run_feature_screen
from routing.data import (
    load_complexity_column,
    load_complexity_from_selection,
    load_oracle,
    merge_tables,
    normalize_signals_csv,
    write_complexity_selection,
)
from routing.compare_generalization import compare_generalization, format_summary
from routing.doctor import run_doctor
from routing.oracle import run_oracle_assessment
from routing.plots import (
    plot_calibration,
    plot_conceptual_model,
    plot_distributions,
    plot_entropy_margin_scatter,
    plot_family_map,
    plot_information_decomposition,
    plot_ladder,
    plot_layer_evolution,
    plot_probability_calibration,
    plot_recovery_matrix,
    plot_roc_curves,
    plot_signal_calibration,
    plot_signal_semantics,
)
from routing.formation_analysis import analyze_formation
from routing.constants import (
    BOOTSTRAP_COUNT,
    BOOTSTRAP_SEED,
    CALIB_REDRAW_FOLDS,
    CALIB_SIZE,
    COL_ENTROPY_WEAK,
    COMPLEMENTARITY_CV_FOLDS,
    DEFAULT_COMPLEXITY_SELECTION_NAME,
    DEFAULT_TOKENIZER_ID,
    MARGIN_TOL_DEFAULT,
    PERMUTATION_COUNT,
    ROUTING_RELEVANCE_SIGNALS,
    STABILITY_CALIB_DRAWS,
)
from routing.splits import (
    load_splits,
    query_ids_for_role,
    resolve_hf_split_and_limit,
    write_policy_splits,
    write_splits,
)
from routing.datasets import load_queries
from routing.model_utils import load_tokenizer
from routing.policies import run_routing_holdout, run_routing_preview


def _query_filter_from_args(args: argparse.Namespace) -> set[str] | None:
    splits_path = getattr(args, "splits_json", None)
    role = getattr(args, "split_role", None)
    if splits_path is None:
        if role:
            raise SystemExit("--split-role requires --splits-json")
        return None
    if not role:
        raise SystemExit("--splits-json requires --split-role calib|test")
    return query_ids_for_role(load_splits(splits_path), role)


def _splits_meta_from_args(args: argparse.Namespace) -> dict | None:
    splits_path = getattr(args, "splits_json", None)
    if splits_path is None:
        return None
    return load_splits(splits_path)


def _resolve_split_load(args: argparse.Namespace) -> tuple[str, int]:
    return resolve_hf_split_and_limit(
        splits=_splits_meta_from_args(args),
        split_role=getattr(args, "split_role", None),
        default_split=getattr(args, "split", "test"),
        default_limit=args.limit,
    )


def _resolve_expected_calib_n(args: argparse.Namespace) -> int:
    """Resolve expected CALIB size for D46 screening."""
    if getattr(args, "expected_n", None) is not None:
        return int(args.expected_n)
    splits_path = getattr(args, "splits_json", None)
    if splits_path is not None:
        split = load_splits(splits_path)
        return int(split.get("calib_size", len(split["calib"])))
    raise SystemExit("screen requires --splits-json or --expected-n for CALIB size")


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=None,
        help="Filter queries by calibration/evaluation ids from split manifest",
    )
    parser.add_argument(
        "--split-role",
        choices=["calib", "test"],
        default=None,
        help="With --splits-json, load the manifest's source split for calib/test role",
    )


def cmd_oracle(args: argparse.Namespace) -> int:
    split, limit = _resolve_split_load(args)
    return run_oracle_assessment(
        weak=args.weak,
        strong=args.strong,
        dataset=args.dataset,
        split=split,
        limit=limit,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        output=args.output,
        strong_only=args.strong_only,
        weak_only=args.weak_only,
        no_resume=args.no_resume,
        query_filter=_query_filter_from_args(args),
        max_pending=getattr(args, "max_pending", None) or None,
    )


def cmd_probes(args: argparse.Namespace) -> int:
    if args.prompt is not None and args.dataset is not None:
        raise SystemExit("Use --prompt or --dataset, not both")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    split, limit = _resolve_split_load(args)
    if getattr(args, "layerwise", False):
        if args.dataset is None:
            raise SystemExit("layerwise requires --dataset")
        if args.prompt is not None:
            raise SystemExit("layerwise is incompatible with --prompt")
        if args.batch_size != 1:
            print("warning: layerwise currently runs batch_size=1 per query", file=sys.stderr)
        return run_layerwise_extraction(
            model=args.model,
            output=args.output,
            trace_path=args.layer_trace,
            dataset=args.dataset,
            split=split,
            limit=limit,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            stab_eps=args.stab_eps,
            stab_k=args.stab_k,
            margin_tol=args.margin_tol,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            query_filter=_query_filter_from_args(args),
            repr_only=getattr(args, "repr_only", False),
        )
    if args.prompt is None and args.dataset is None:
        raise SystemExit("Provide --prompt or --dataset")
    return run_probe_extraction(
        model=args.model,
        output=args.output,
        prompt=args.prompt,
        query_id=args.query_id,
        dataset=args.dataset,
        split=split,
        limit=limit,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        query_filter=_query_filter_from_args(args),
    )


def cmd_features(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer_model)
    tokenizer_id = getattr(tokenizer, "name_or_path", args.tokenizer_model)
    print(f"tokenizer: {tokenizer_id}  (no model forward pass)")

    split, limit = _resolve_split_load(args)
    queries = load_queries(args.dataset, split, limit, args.seed)
    qf = _query_filter_from_args(args)
    if qf is not None:
        from routing.datasets import filter_queries
        queries = filter_queries(queries, qf)
        print(f"split filter: {len(queries)} queries")

    return run_feature_extraction(
        queries=queries,
        tokenizer=tokenizer,
        output=args.output,
        tokenizer_id=tokenizer_id,
        mattr_window=args.mattr_window,
    )


def cmd_screen(args: argparse.Namespace) -> int:
    feats = pd.read_csv(args.features)
    oracle = load_oracle(args.oracle, include_gap=False)
    frame = feats.merge(oracle[["query_id", "y_opp", "bucket"]], on="query_id", how="inner")
    if frame.empty:
        raise SystemExit("No rows after merge — check query_id alignment.")

    try:
        payload = run_feature_screen(
            frame,
            expected_n=_resolve_expected_calib_n(args),
            allow_preview=args.allow_preview,
            boot_count=args.bootstrap_n,
            boot_seed=args.bootstrap_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    selected = payload["selection"]["selected_feature"]
    print(f"\nRepresentative feature: {selected!r} (score={payload['selection']['composite_score']:.4f})")
    if not payload["calib_locked"]:
        print("WARNING: preview run — do not lock D46.")
    else:
        selection_path = args.selection_output or (args.output.parent / DEFAULT_COMPLEXITY_SELECTION_NAME)
        write_complexity_selection(
            selection_path,
            selected_feature=selected,
            calib_locked=payload["calib_locked"],
            n=payload["n"],
            screen_report=args.output,
            features_path=args.features,
            oracle_path=args.oracle,
        )
        print(f"Wrote frozen selection {selection_path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    weak = normalize_signals_csv(args.weak_csv)
    strong = normalize_signals_csv(args.strong_csv)
    oracle = load_oracle(args.oracle)

    query_features = None
    if args.features_csv is not None:
        if args.complexity_selection is not None:
            query_features = load_complexity_from_selection(args.features_csv, args.complexity_selection)
        elif args.c_q_column:
            if not args.allow_manual_c_q:
                raise SystemExit(
                    "Use --complexity-selection (frozen D46 from screen) instead of --c-q-column. "
                    "Pass --allow-manual-c-q only for dev/smoke runs."
                )
            query_features = load_complexity_column(args.features_csv, source_column=args.c_q_column)
        else:
            raise SystemExit(
                "When --features-csv is set, provide --complexity-selection "
                f"(from screen → {DEFAULT_COMPLEXITY_SELECTION_NAME})"
            )

    merged = merge_tables(
        weak,
        strong,
        oracle,
        query_features=query_features,
        skip_prompt_hash_verify=getattr(args, "skip_prompt_hash_verify", False),
    )
    analysis = analyze_routing_relevance(
        merged, args.oracle, n_boot=args.bootstrap_n, bootstrap_seed=args.bootstrap_seed
    )
    summary = build_relevance_summary(
        merged=merged,
        weak_path=args.weak_csv,
        strong_path=args.strong_csv,
        oracle_path=args.oracle,
        analysis=analysis,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}  (n={len(merged)})")

    if args.merged_csv:
        args.merged_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.merged_csv, index=False)
        print(f"Wrote {args.merged_csv}")

    print("\nHeadline signals vs routing opportunity (Spearman rho, bootstrap CI):")
    for name in ROUTING_RELEVANCE_SIGNALS:
        m = analysis["correlation_vs_opportunity"].get(name, {})
        rho = m.get("spearman_rho")
        lo, hi = m.get("spearman_ci_low"), m.get("spearman_ci_high")
        if rho is not None:
            ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None and hi is not None else "[CI n/a]"
            print(f"  {name:18s}  rho={rho:+.3f}  {ci}")
    return 0


def cmd_complementarity(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.merged_csv)
    calib_ids = test_ids = None
    eval_mode = args.eval_mode

    if args.splits_json:
        split = load_splits(args.splits_json)
        calib_ids = split["calib"]
        test_ids = split["test"]
        eval_mode = "holdout"
    elif not args.allow_cv:
        raise SystemExit(
            "Study III requires nested evaluation: pass --splits-json "
            "(fit logistic on CALIB, AUROC on TEST). Dev/smoke only: --allow-cv"
        )
    elif eval_mode == "holdout":
        raise SystemExit("holdout eval requires --splits-json")

    if eval_mode == "holdout":
        have = set(frame["query_id"].astype(str))
        missing_calib = calib_ids - have
        missing_test = test_ids - have
        if missing_calib or missing_test:
            raise SystemExit(
                "merged CSV must include both CALIB and TEST rows for nested eval. "
                f"missing calib={len(missing_calib)} test={len(missing_test)} — "
                "extract on both roles (validation + test) into one oracle/CSV set."
            )

    perm_n = 0 if args.no_permutation else args.permutation_n
    calib_folds = 1 if args.no_calib_redraw else args.calib_redraw_folds

    result = run_complementarity_analysis(
        frame,
        boot_count=args.bootstrap_n,
        boot_seed=args.bootstrap_seed,
        cv_folds=args.cv_folds,
        eval_mode=eval_mode,
        calib_ids=calib_ids,
        test_ids=test_ids,
        permutation_n=perm_n,
        permutation_seed=args.permutation_seed,
        allow_cv=args.allow_cv,
        calib_redraw_folds=calib_folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")

    ladder = result.get("ablation_ladder", {}).get("models", [])
    if ladder:
        print("\nAblation ladder (Study III centerpiece):")
        for m in ladder:
            ci = ""
            if m.get("auroc_ci_low") is not None:
                ci = f" [{m['auroc_ci_low']:.3f}, {m['auroc_ci_high']:.3f}]"
            print(f"  {m['label']:20s}  AUROC={m.get('auroc')}{ci}")

    pe = result.get("primary_endpoint", {})
    if pe:
        print(
            f"\nPrimary endpoint ΔAUROC c→c+H+m: {pe.get('delta_auroc'):+.3f} "
            f"[{pe.get('ci_low')}, {pe.get('ci_high')}]  "
            f"DeLong p={pe.get('delong_p')}  perm p={pe.get('permutation_p')}"
        )
        rcs = pe.get("repeated_calib_summary")
        if rcs and rcs.get("mean") is not None:
            print(
                f"  5-fold CALIB redraw: ΔAUROC mean={rcs['mean']:+.3f} ± {rcs['std']:.3f}"
            )
    ra = result.get("rank_agreement", {}).get("complexity_entropy", {})
    if ra.get("spearman_rho") is not None:
        print(
            f"Rank agreement c vs H: Spearman ρ={ra['spearman_rho']:+.3f}  "
            f"Kendall τ={ra.get('kendall_tau')}"
        )
    steps = result.get("auc_increments", {})
    step = steps.get("steps", steps) if isinstance(steps, dict) else {}
    step = step.get("entropy_beyond_complexity", {}) if isinstance(step, dict) else {}
    print(
        f"\nΔAUROC c→c+H: {step.get('delta_auroc'):+.3f} "
        f"[{step.get('ci_low')}, {step.get('ci_high')}]"
    )
    perm = result.get("permutation_tests", {})
    tests = perm.get("tests", perm) if isinstance(perm, dict) else {}
    if tests:
        print("\nPermutation p-values (supporting):")
        for name, pt in tests.items():
            print(f"  {name:18s}  p={pt.get('p_value')}")
    return 0


def cmd_failure_analysis(args: argparse.Namespace) -> int:
    from routing.constants import PROBE_DERIVED_MARGIN

    df = pd.read_csv(args.merged_csv)
    if args.signal == "both":
        result = run_failure_analysis_bundle(df, n=args.n)
    elif args.signal == "entropy":
        result = run_failure_analysis(df, signal_col=COL_ENTROPY_WEAK, n=args.n)
    else:
        result = run_failure_analysis(df, signal_col=PROBE_DERIVED_MARGIN, n=args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}  (n={args.n} per tail, signal={args.signal})")
    return 0


def cmd_interpret(args: argparse.Namespace) -> int:
    merged_full = None
    test_ids = None
    calib_ids = None
    if args.splits_json:
        splits = load_splits(args.splits_json)
        test_ids = splits["test"]
        calib_ids = splits["calib"]
        full_path = args.merged_full_csv
        if full_path is None:
            candidate = args.merged_csv.parent / "arc_merged_full.csv"
            full_path = candidate if candidate.exists() else None
        if full_path is not None:
            merged_full = pd.read_csv(full_path)
    result = run_interpretation_analysis(
        pd.read_csv(args.merged_csv),
        features_csv=args.features_csv,
        complementarity_json=args.complementarity_json,
        merged_full=merged_full,
        test_ids=test_ids,
        calib_ids=calib_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}  (n={result['n']})")
    landscape = result["routing_landscape"]["counts"]
    print(f"  buckets: {landscape}")
    gap = result["oracle_gap"]["correlations"].get("delta_margin_gain", {})
    if gap.get("spearman_rho") is not None:
        print(f"  oracle_gap ρ(Δm_gain) = {gap['spearman_rho']:+.3f}")
    reg = result["entropy_regression"]
    if reg.get("r_squared") is not None:
        print(f"  entropy ~ features R² = {reg['r_squared']:.3f}")
    cal = result.get("signal_calibration", {}).get("entropy_w", {}).get("bins", {})
    if cal:
        print("  H_w quantile calibration (opportunity rate):")
        for band in ("bottom", "middle", "top"):
            b = cal[band]
            print(f"    {band:6s}  {b['opportunity_rate']:.1%}  (n={b['n']})")
    prob = result.get("probability_calibration", {}).get("complexity_joint", {})
    if prob.get("bins"):
        print(f"  logistic calibration ({prob.get('calibration_verdict', 'n/a')}, mean gap {prob.get('mean_signed_gap', 0):+.3f}):")
        for b in prob["bins"]:
            print(
                f"    bin {b['bin']:2d}  pred={b['mean_predicted']:.2f}  "
                f"obs={b['observed_rate']:.2f}  gap={b['gap']:+.2f}"
            )
    igap = result.get("information_gap", {})
    if igap.get("accuracies"):
        acc = igap["accuracies"]
        gaps = igap["gaps_percentage_points"]
        print(
            f"  information gap: oracle {acc['oracle']:.1%} | "
            f"learned {acc['learned_router']:.1%} | "
            f"headroom {gaps['routing_headroom']:.1%} | "
            f"exploited {gaps['exploited_by_learned']:.1%}"
        )
    return 0


def cmd_stability(args: argparse.Namespace) -> int:
    split = load_splits(args.splits_json)
    result = run_calibration_stability(
        pd.read_csv(args.merged_csv),
        test_ids=split["test"],
        calib_size=args.calib_size,
        n_draws=args.n_draws,
        base_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(f"Wrote {args.output}")
    print(f"\nJoint model TEST AUROC: {s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}")
    for col, stats in s["coefficients"].items():
        print(f"  β({col}) = {stats['mean']:+.3f} ± {stats['std']:.3f}")
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    ds = args.dataset.lower().replace("-", "_")
    if not args.legacy_subset:
        meta = write_policy_splits(args.output, dataset=ds, seed=args.seed)
        print(
            f"Wrote {args.output}  protocol={meta['protocol']}  "
            f"dataset={meta['dataset']}  policy={meta['policy']}  "
            f"calib={len(meta['calib'])} ({meta['calib_split']})  "
            f"test={len(meta['test'])} ({meta['test_split']})"
        )
        return 0

    from routing.splits import draw_calib_ids

    queries = load_queries(args.dataset, args.split, args.limit, args.seed)
    all_ids = [q["id"] for q in queries]
    calib_set = draw_calib_ids(all_ids, calib_size=args.calib_size, seed=args.seed)
    test_ids = [qid for qid in all_ids if qid not in calib_set]
    write_splits(
        args.output,
        calib=sorted(calib_set),
        test=test_ids,
        seed=args.seed,
        dataset=ds,
        protocol="internal_random_subset_v1",
        policy="internal_random_subset",
        calib_split=args.split,
        test_split=args.split,
    )
    print(
        f"Wrote {args.output}  protocol=random_subset  "
        f"calib={len(calib_set)} test={len(test_ids)}"
    )
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    if args.figure == "distributions":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for distributions")
        plot_distributions(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "roc":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for roc")
        plot_roc_curves(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "scatter":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for scatter")
        plot_entropy_margin_scatter(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "family-map":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for family-map")
        plot_family_map(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "ladder":
        if not args.complementarity_json:
            raise SystemExit("--complementarity-json required for ladder")
        plot_ladder(args.complementarity_json, args.output, dpi=args.dpi)
    elif args.figure == "entropy-calibration":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for entropy-calibration")
        plot_signal_calibration(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "probability-calibration":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required (use arc_merged_full.csv)")
        if not args.splits_json:
            raise SystemExit("--splits-json required for probability-calibration")
        plot_probability_calibration(args.merged_csv, args.splits_json, args.output, dpi=args.dpi)
    elif args.figure == "calibration":
        if not args.complementarity_json:
            raise SystemExit("--complementarity-json required for calibration")
        plot_calibration(args.complementarity_json, args.output, dpi=args.dpi)
    elif args.figure == "decomposition":
        if not args.complementarity_json:
            raise SystemExit("--complementarity-json required for decomposition")
        plot_information_decomposition(args.complementarity_json, args.output, dpi=args.dpi)
    elif args.figure == "semantics":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for semantics")
        plot_signal_semantics(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "recovery-matrix":
        if not args.merged_csv:
            raise SystemExit("--merged-csv required for recovery-matrix")
        plot_recovery_matrix(args.merged_csv, args.output, dpi=args.dpi)
    elif args.figure == "conceptual-model":
        plot_conceptual_model(args.output, dpi=args.dpi)
    elif args.figure == "formation":
        if not args.layer_trace or not args.merged_csv:
            raise SystemExit("--layer-trace and --merged-csv required for formation")
        plot_layer_evolution(
            args.layer_trace,
            args.merged_csv,
            args.output,
            dpi=args.dpi,
            trace_metric=getattr(args, "trace_metric", "margin"),
        )
    print(f"Wrote {args.output}")
    return 0


def cmd_analyze_formation(args: argparse.Namespace) -> int:
    payload = analyze_formation(
        trace_path=args.layer_trace,
        merged_csv=args.merged_csv,
        output=args.output,
        trace_metric=getattr(args, "trace_metric", "margin"),
    )
    print(payload.get("interpretation", ""))
    print(f"Wrote {args.output}")
    return 0


def cmd_layerwise_parity(args: argparse.Namespace) -> int:
    split, limit = _resolve_split_load(args)
    return run_terminal_parity_smoke(
        model=args.model,
        dataset=args.dataset,
        split=split,
        limit=limit,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        margin_tol=args.margin_tol,
        query_filter=_query_filter_from_args(args),
    )


def cmd_route_preview(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.merged_csv)
    required = {"weak_ok", "strong_ok", "bucket", "entropy_w", "margin_w", "entropy_s", "delta_entropy"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"merged CSV missing: {missing}")

    summary = run_routing_preview(df, weak_cost=args.weak_cost, strong_cost=args.strong_cost)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}\n")

    print(f"{'Policy':<28} {'Acc':>6} {'Cost':>6} {'Opp→Strong':>11} {'Weak-only OK':>12}")
    print("-" * 70)
    for r in summary["policies"]:
        opp = r["opportunity_recall_strong"]
        wo = r["weak_only_correct_routing"]
        opp_s = f"{opp:.0%}" if opp is not None else "n/a"
        wo_s = f"{wo:.0%}" if wo is not None else "n/a"
        print(f"{r['policy']:<28} {r['accuracy']:>6.1%} {r['avg_cost']:>6.2f} {opp_s:>11} {wo_s:>12}")
    return 0


def cmd_route_eval(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.merged_csv)
    required = {
        "query_id", "weak_ok", "strong_ok", "bucket", "y_opp",
        "c_q", "entropy_w", "margin_w",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"merged CSV missing: {missing}")

    if not args.splits_json:
        raise SystemExit(
            "Study IV requires nested evaluation: pass --splits-json "
            "(fit + tune on CALIB, report on TEST)."
        )
    split = load_splits(args.splits_json)
    calib_ids = split["calib"]
    test_ids = split["test"]

    summary = run_routing_holdout(
        df,
        calib_ids=calib_ids,
        test_ids=test_ids,
        weak_cost=args.weak_cost,
        strong_cost=args.strong_cost,
        cost_lambda=args.cost_lambda,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}\n")
    print(f"Router τ={summary['router']['tau']:.4f}  (fit/tune CALIB, evaluate TEST n={summary['nested_evaluation']['n_test']})\n")
    print(f"{'Policy':<18} {'Acc':>6} {'Cost':>6} {'Opp→Strong':>11} {'Weak-only OK':>12}")
    print("-" * 58)
    for r in summary["policies_test"]:
        opp = r.get("opportunity_recall_strong")
        wo = r.get("weak_only_correct_routing")
        opp_s = f"{opp:.0%}" if opp is not None else "n/a"
        wo_s = f"{wo:.0%}" if wo is not None else "n/a"
        print(f"{r['policy']:<18} {r['accuracy']:>6.1%} {r['avg_cost']:>6.2f} {opp_s:>11} {wo_s:>12}")
    return 0


def cmd_compare_generalization(args: argparse.Namespace) -> int:
    regimes: list[tuple[str, Path]] = []
    for spec in args.regime:
        if "=" not in spec:
            raise SystemExit(f"Expected NAME=PATH for --regime, got {spec!r}")
        name, path = spec.split("=", 1)
        regimes.append((name.strip(), Path(path.strip())))

    payload = compare_generalization(
        regimes=regimes,
        output=args.output,
        mmlu_subject_splits=not args.no_mmlu_subjects,
        n_boot=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(format_summary(payload))
    if args.output:
        print(f"\nWrote {args.output}")
    return 0


def cmd_summarize_c2(args: argparse.Namespace) -> int:
    summary = summarize_c2(args.oracle, scientific_question=args.scientific_question)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(
        oracle=args.oracle,
        weak_csv=args.weak_csv,
        strong_csv=args.strong_csv,
        merged_csv=args.merged_csv,
        features_csv=args.features_csv,
        complexity_selection=args.complexity_selection,
        tokenizer_model=args.tokenizer_model,
        strict=args.strict,
    )
    payload = report.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")

    icons = {"pass": "✓", "warn": "!", "fail": "✗", "skip": "-"}
    for check in report.checks:
        icon = icons.get(check.status, "?")
        detail = f" — {check.detail}" if check.detail else ""
        print(f"  {icon} {check.name}{detail}")

    if not report.ok:
        print("\nFAILED — fix issues above before scaling runs.")
        return 1
    print("\nOK — artifacts look consistent.")
    return 0


def cmd_verify_logprobs(args: argparse.Namespace) -> int:
    def peak_rss_mb() -> float:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    load_kwargs = {"revision": args.revision} if args.revision else {}

    print(f"model_id: {args.model_id}")
    print(f"revision: {args.revision or 'main (default)'}")
    print(f"dtype: {args.dtype}")
    print(f"torch: {torch.__version__}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto", **load_kwargs
    )
    load_s = time.perf_counter() - t0

    inputs = tokenizer(args.prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t1 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    forward_s = time.perf_counter() - t1

    logits = outputs.logits
    logits_vocab = logits.shape[-1]
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"logits vocab dim: {logits_vocab}")
    print(f"load time (s): {load_s:.1f}")
    print(f"forward time (s): {forward_s:.3f}")
    print(f"peak rss (MB): {peak_rss_mb():.0f}")

    if model.config.vocab_size != logits_vocab:
        print(
            f"NOTE: config.vocab_size ({model.config.vocab_size}) "
            f"!= logits dim ({logits_vocab})"
        )

    if logits_vocab > 1000:
        print("PASS: full-vocabulary logits returned")
        return 0
    print("CHECK: unexpected vocab size — inspect architecture")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("oracle", help="Routing opportunity assessment (C2)")
    p.add_argument("--weak", required=True)
    p.add_argument("--strong", required=True)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--strong-only", action="store_true")
    p.add_argument("--weak-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="Process at most N not-yet-cached queries per model pass (0=all pending; use for CPU batches)",
    )
    _add_split_args(p)
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("probes", help="Prefill probe extraction")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--query-id", default="smoke_0")
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--layerwise", action="store_true", help="C3 layerwise margin trajectories")
    p.add_argument(
        "--repr-only",
        action="store_true",
        dest="repr_only",
        help="Skip intermediate LM-head probes; terminal margin + Route B drift only (faster re-extract)",
    )
    p.add_argument("--layer-trace", type=Path, default=None, dest="layer_trace", help="JSONL trace output")
    p.add_argument("--stab-eps", type=float, default=0.02, dest="stab_eps")
    p.add_argument("--stab-k", type=int, default=2, dest="stab_k")
    p.add_argument("--margin-tol", type=float, default=MARGIN_TOL_DEFAULT, dest="margin_tol", help="Terminal margin parity tolerance (layerwise smoke)")
    p.add_argument("--overwrite", action="store_true", help="Replace existing --output / --layer-trace (layerwise)")
    _add_split_args(p)
    p.set_defaults(func=cmd_probes)

    p = sub.add_parser("features", help="Query complexity features")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mattr-window", type=int, default=15)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_ID)
    _add_split_args(p)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("screen", help="D46 feature screening on CALIB")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--selection-output",
        type=Path,
        default=None,
        help=f"Frozen D46 artifact (default: same dir as --output / {DEFAULT_COMPLEXITY_SELECTION_NAME})",
    )
    p.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help="Expected CALIB size (default: from --splits-json calib_size)",
    )
    p.add_argument(
        "--splits-json",
        type=Path,
        default=None,
        help="Optional split manifest to auto-resolve expected CALIB size",
    )
    p.add_argument("--allow-preview", action="store_true")
    p.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_COUNT)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("merge", help="Merge signals and compute routing relevance")
    p.add_argument("--weak-csv", type=Path, required=True)
    p.add_argument("--strong-csv", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--merged-csv", type=Path, default=None)
    p.add_argument("--features-csv", type=Path, default=None)
    p.add_argument(
        "--complexity-selection",
        type=Path,
        default=None,
        help=f"Frozen D46 selection from screen ({DEFAULT_COMPLEXITY_SELECTION_NAME})",
    )
    p.add_argument("--c-q-column", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--allow-manual-c-q",
        action="store_true",
        help="Dev only: use --c-q-column instead of frozen selection",
    )
    p.add_argument(
        "--skip-prompt-hash-verify",
        action="store_true",
        help="Skip oracle vs probe prompt_hash check (dev only)",
    )
    p.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_COUNT)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("complementarity", help="Study III — nested CALIB-fit / TEST-eval")
    p.add_argument("--merged-csv", type=Path, required=True,
                   help="Must contain both CALIB and TEST query_ids")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits-json", type=Path, default=None,
                   help="CALIB/TEST manifest (required unless --allow-cv)")
    p.add_argument(
        "--eval-mode",
        choices=["holdout", "cv"],
        default="holdout",
        help="holdout=default (CALIB fit, TEST eval); cv=dev only with --allow-cv",
    )
    p.add_argument(
        "--allow-cv",
        action="store_true",
        help="Dev/smoke only: in-sample CV without splits (not for paper)",
    )
    p.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_COUNT)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument(
        "--cv-folds",
        type=int,
        default=COMPLEMENTARITY_CV_FOLDS,
        help="Stratified k-fold CV when eval-mode=cv",
    )
    p.add_argument("--permutation-n", type=int, default=PERMUTATION_COUNT)
    p.add_argument("--permutation-seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--no-permutation", action="store_true", help="Skip permutation tests (dev)")
    p.add_argument(
        "--calib-redraw-folds",
        type=int,
        default=CALIB_REDRAW_FOLDS,
        help="CALIB redraw folds with fixed TEST (fold 0 = canonical split)",
    )
    p.add_argument(
        "--no-calib-redraw",
        action="store_true",
        help="Skip repeated CALIB folds (fold 0 only; faster dev)",
    )
    p.set_defaults(func=cmd_complementarity)

    p = sub.add_parser("stability", help="CALIB redraw stability (appendix)")
    p.add_argument("--merged-csv", type=Path, required=True)
    p.add_argument("--splits-json", type=Path, required=True, help="Fixed TEST ids")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calib-size", type=int, default=CALIB_SIZE)
    p.add_argument("--n-draws", type=int, default=STABILITY_CALIB_DRAWS)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p.set_defaults(func=cmd_stability)

    p = sub.add_parser("splits", help="Write CALIB/TEST split manifest using dataset policy")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", help="Legacy internal split mode only")
    p.add_argument("--limit", type=int, default=1172)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-size", type=int, default=CALIB_SIZE)
    p.add_argument(
        "--legacy-subset",
        action="store_true",
        help="Bypass dataset policy and create internal CALIB/TEST from one split",
    )
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("failure-analysis", help="Top/bottom signal tails for Discussion")
    p.add_argument("--merged-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument(
        "--signal",
        choices=["both", "entropy", "delta_margin"],
        default="both",
        help="Signal for tail analysis (default: both H_w and Δm_gain)",
    )
    p.set_defaults(func=cmd_failure_analysis)

    p = sub.add_parser("interpret", help="Q2–Q3 interpretation bundle (landscape, overlap, decomposition)")
    p.add_argument("--merged-csv", type=Path, required=True)
    p.add_argument("--features-csv", type=Path, default=None, help="Query features for entropy regression")
    p.add_argument("--complementarity-json", type=Path, default=None, help="Study III JSON for decomposition block")
    p.add_argument("--splits-json", type=Path, default=None, help="CALIB/TEST split for logistic probability calibration")
    p.add_argument("--merged-full-csv", type=Path, default=None, help="CALIB+TEST merge (default: arc_merged_full.csv)")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_interpret)

    p = sub.add_parser("plot", help="Generate paper figures")
    p.add_argument(
        "figure",
        choices=["distributions", "roc", "scatter", "ladder", "family-map", "calibration", "decomposition", "entropy-calibration", "probability-calibration", "semantics", "recovery-matrix", "conceptual-model", "formation"],
    )
    p.add_argument("--merged-csv", type=Path, default=None)
    p.add_argument("--layer-trace", type=Path, default=None, dest="layer_trace")
    p.add_argument(
        "--trace-metric",
        choices=["margin", "drift"],
        default="margin",
        dest="trace_metric",
        help="Curve metric for formation plot / analyze-formation (margin=Route A, drift=Route B)",
    )
    p.add_argument("--complementarity-json", type=Path, default=None)
    p.add_argument("--splits-json", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=150)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("analyze-formation", help="RH5 divergence + bucket medians from layer traces")
    p.add_argument("--layer-trace", type=Path, required=True, dest="layer_trace")
    p.add_argument("--merged-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--trace-metric",
        choices=["margin", "drift"],
        default="margin",
        dest="trace_metric",
        help="margin=Route A (logit-lens); drift=Route B (adjacent repr drift)",
    )
    p.set_defaults(func=cmd_analyze_formation)

    p = sub.add_parser(
        "layerwise-parity",
        help="Pre-TEST smoke: verify lm_head(last_hidden_state) matches out.logits",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default=None)
    p.add_argument("--margin-tol", type=float, default=MARGIN_TOL_DEFAULT, dest="margin_tol")
    _add_split_args(p)
    p.set_defaults(func=cmd_layerwise_parity)

    p = sub.add_parser("route-preview", help="Median-threshold routing sanity check")
    p.add_argument("--merged-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weak-cost", type=float, default=1.0)
    p.add_argument("--strong-cost", type=float, default=3.0)
    p.set_defaults(func=cmd_route_preview)

    p = sub.add_parser(
        "route-eval",
        help="Study IV hold-out routing (CALIB fit + threshold, TEST eval)",
    )
    p.add_argument("--merged-csv", type=Path, required=True)
    p.add_argument("--splits-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weak-cost", type=float, default=1.0)
    p.add_argument("--strong-cost", type=float, default=3.0)
    p.add_argument(
        "--cost-lambda",
        type=float,
        default=0.0,
        help="CALIB threshold objective: accuracy - lambda*avg_cost (0 = max accuracy)",
    )
    p.set_defaults(func=cmd_route_eval)

    p = sub.add_parser("compare-generalization", help="RH7 dimension transfer across merged regimes")
    p.add_argument(
        "--regime",
        action="append",
        required=True,
        metavar="NAME=MERGED_CSV",
        help="Regime label and merged CSV (repeatable, e.g. arc=analysis/arc_merged.csv)",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--no-mmlu-subjects", action="store_true", help="Skip per-subject MMLU breakdown")
    p.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_COUNT)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.set_defaults(func=cmd_compare_generalization)

    p = sub.add_parser("summarize-c2", help="Summarize C2 oracle JSON")
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scientific-question", default=None)
    p.set_defaults(func=cmd_summarize_c2)

    p = sub.add_parser("verify-logprobs", help="Feasibility check for full-vocab logits")
    p.add_argument("model_id")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--prompt", default="User: What is 2+2?\nAssistant:")
    p.set_defaults(func=cmd_verify_logprobs)

    p = sub.add_parser("doctor", help="Pre-flight checks on experiment artifacts")
    p.add_argument("--oracle", type=Path, default=None)
    p.add_argument("--weak-csv", type=Path, default=None)
    p.add_argument("--strong-csv", type=Path, default=None)
    p.add_argument("--merged-csv", type=Path, default=None)
    p.add_argument("--features-csv", type=Path, default=None)
    p.add_argument("--complexity-selection", type=Path, default=None)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_ID)
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
