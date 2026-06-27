"""Layerwise prefill probes — margin trajectories and formation scalars (C3 / RH5).

Primary scalar: ``stabilization_layer``. ``slope_margin`` is secondary.

Probe statistics use raw ``softmax(logits)`` at each layer — the model's own
output space. No logit temperature rescaling (document as limitation if asked).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from routing.constants import (
    MARGIN_TOL_DEFAULT,
    PROBE_LAYERWISE_FIELDS,
    PROBE_METHOD_LAYERWISE,
    STAB_EPS_DEFAULT,
    STAB_K_DEFAULT,
)
from routing.datasets import filter_queries, load_queries
from routing.model_dependent import ProbeMetrics, extract_logits, make_probe_row
from routing.model_utils import load_model_and_tokenizer, release_model, resolve_input_device
from routing.prompt_protocol import build_chat_prompt


@dataclass(frozen=True)
class LayerTrace:
    num_layers: int
    margins: list[float]
    entropies: list[float]
    slope_margin: float
    stabilization_layer: int


@dataclass(frozen=True)
class TerminalParityResult:
    query_id: str
    margin_from_logits: float
    margin_from_logit_lens: float
    margin_delta: float
    max_logit_delta: float
    passed: bool


def depth_fraction_list(num_layers: int) -> list[float]:
    """Normalized depth ℓ/L for each layer (1-indexed ℓ), length L."""
    if num_layers <= 0:
        return []
    return [(i + 1) / num_layers for i in range(num_layers)]


def slope_margin(margins: list[float]) -> float:
    """OLS slope of m_ℓ vs ℓ — supplementary scalar only (not headline RH5)."""
    n = len(margins)
    if n < 2:
        return 0.0
    x = np.arange(1, n + 1, dtype=float)
    return float(np.polyfit(x, np.asarray(margins, dtype=float), 1)[0])


def stab_layer(margins: list[float], eps: float, k: int) -> int:
    """Smallest 1-indexed stabilization layer (Methods definition below).

    Stabilization requires observing at least one adjacent margin step. When
    ``L >= 2``, the minimum possible value is 2; layer 1 cannot stabilize
    because there is no prior margin to compare against.

    ``transitions[i] = |margins[i + 1] - margins[i]|``.  A run of ``k``
    consecutive stable transitions starting at index ``start`` means
    ``transitions[start:start+k]`` are all ``< eps``.  Return ``start + 2``:
    the **first layer immediately following** the beginning of that run
    (e.g. stable transition 1→2 ⇒ ``start=0`` ⇒ stabilization layer 2).
    If none, or ``L < 2``, return ``L``.
    """
    n = len(margins)
    if n == 0:
        return 0
    if n == 1:
        return 1  # L; no transition to observe — stabilization not defined
    k = max(1, k)

    transitions = [abs(margins[i + 1] - margins[i]) for i in range(n - 1)]
    for start in range(len(transitions)):
        if start + k > len(transitions):
            break
        if all(t < eps for t in transitions[start : start + k]):
            return start + 2  # minimum 2 when L >= 2
    return n


def formation_scalars(margins: list[float], eps: float, k: int) -> tuple[float, int]:
    return slope_margin(margins), stab_layer(margins, eps, k)


def _prepare_output_path(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"{path} already exists — pass overwrite=True or --overwrite to replace"
        )
    path.unlink()


def _final_norm(model):
    inner = getattr(model, "model", None)
    norm = getattr(inner, "norm", None) if inner is not None else None
    if norm is None:
        raise RuntimeError(
            "terminal parity smoke requires model.model.norm (Llama-style architecture)"
        )
    return norm


def verify_terminal_logit_parity(
    model,
    tokenizer,
    user_content: str,
    *,
    query_id: str = "smoke",
    margin_tol: float = MARGIN_TOL_DEFAULT,
) -> TerminalParityResult:
    """Smoke: logit-lens terminal path must match out.logits at last prompt token."""
    built = build_chat_prompt(tokenizer, user_content)
    device = resolve_input_device(model)
    inputs = tokenizer(built["chat_prompt"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)

    batch_idx = 0
    last_pos = int(inputs["attention_mask"].sum(dim=1).item() - 1)
    hs = out.hidden_states
    n_layers = int(model.config.num_hidden_layers)
    if hs is None or len(hs) < n_layers + 1:
        raise RuntimeError(
            f"hidden_states missing layers (got {len(hs) if hs else 0}, need {n_layers + 1})"
        )

    logits_ref = out.logits[batch_idx, last_pos]
    # Same index as _layer_logits final layer (not hs[-1] — tuple may include post-norm).
    h_pre = hs[n_layers][batch_idx, last_pos, :]
    logits_lens = model.lm_head(_final_norm(model)(h_pre))

    margin_ref = extract_logits(logits_ref, tokenizer).margin
    margin_lens = extract_logits(logits_lens, tokenizer).margin
    margin_delta = abs(margin_ref - margin_lens)
    max_logit_delta = float((logits_ref - logits_lens).abs().max().item())
    passed = margin_delta <= margin_tol
    return TerminalParityResult(
        query_id=query_id,
        margin_from_logits=margin_ref,
        margin_from_logit_lens=margin_lens,
        margin_delta=margin_delta,
        max_logit_delta=max_logit_delta,
        passed=passed,
    )


def run_terminal_parity_smoke(
    *,
    model: str,
    dataset: str,
    split: str,
    limit: int,
    seed: int,
    device: str,
    dtype: str | None,
    margin_tol: float = MARGIN_TOL_DEFAULT,
    query_filter: set[str] | None = None,
) -> int:
    """Pre-TEST gate: verify logit-lens terminal path matches model logits on N queries."""
    queries = load_queries(dataset, split, limit, seed)
    if query_filter is not None:
        queries = filter_queries(queries, query_filter)
    if not queries:
        raise SystemExit("no queries for parity smoke")

    model_obj, tokenizer, device_res = load_model_and_tokenizer(model, device=device, dtype=dtype)
    print(f"layerwise parity smoke  device={device_res}  n={len(queries)}  margin_tol={margin_tol}")

    failures: list[TerminalParityResult] = []
    deltas: list[float] = []
    for q in queries:
        result = verify_terminal_logit_parity(
            model_obj,
            tokenizer,
            q["user_content"],
            query_id=str(q["id"]),
            margin_tol=margin_tol,
        )
        deltas.append(result.margin_delta)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.query_id}: "
            f"margin logits={result.margin_from_logits:.6f} lens={result.margin_from_logit_lens:.6f} "
            f"Δm={result.margin_delta:.6f} max|Δlogit|={result.max_logit_delta:.6f}"
        )
        if not result.passed:
            failures.append(result)

    release_model(model_obj)
    if deltas:
        print(
            f"margin Δ summary: max={max(deltas):.6e}  mean={sum(deltas) / len(deltas):.6e}  "
            f"(tol={margin_tol})"
        )
    if failures:
        print(f"parity FAILED ({len(failures)}/{len(queries)} queries)")
        return 1
    print(f"parity PASSED ({len(queries)} queries)")
    print("PASS: terminal lm_head(norm(h[-1])) matches out.logits")
    return 0


def _layer_logits(
    model,
    hidden_states: tuple,
    batch_idx: int,
    last_pos: int,
    layer_idx: int,
    n_layers: int,
) -> torch.Tensor:
    h = hidden_states[layer_idx + 1][batch_idx, last_pos, :]
    if layer_idx == n_layers - 1:
        h = model.model.norm(h)
    return model.lm_head(h)


def probe_layerwise(
    model,
    tokenizer,
    user_content: str,
    *,
    stab_eps: float = STAB_EPS_DEFAULT,
    stab_k: int = STAB_K_DEFAULT,
    margin_tol: float = MARGIN_TOL_DEFAULT,
) -> tuple[ProbeMetrics, dict[str, Any], LayerTrace]:
    built = build_chat_prompt(tokenizer, user_content)
    device = resolve_input_device(model)
    inputs = tokenizer(built["chat_prompt"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)

    batch_idx = 0
    last_pos = int(inputs["attention_mask"].sum(dim=1).item() - 1)
    terminal = extract_logits(out.logits[batch_idx, last_pos], tokenizer)
    n_layers = int(model.config.num_hidden_layers)
    hs = out.hidden_states
    if hs is None or len(hs) < n_layers + 1:
        raise RuntimeError(f"hidden_states missing layers (got {len(hs) if hs else 0}, need {n_layers + 1})")

    margins: list[float] = []
    entropies: list[float] = []
    for i in range(n_layers):
        metrics = extract_logits(_layer_logits(model, hs, batch_idx, last_pos, i, n_layers), tokenizer)
        margins.append(metrics.margin)
        entropies.append(metrics.entropy)

    if abs(margins[-1] - terminal.margin) > margin_tol:
        raise RuntimeError(
            f"terminal margin mismatch: layerwise={margins[-1]:.6f} logits={terminal.margin:.6f} "
            f"(tol={margin_tol})"
        )

    slope, stab = formation_scalars(margins, stab_eps, stab_k)
    trace = LayerTrace(
        num_layers=n_layers,
        margins=margins,
        entropies=entropies,
        slope_margin=slope,
        stabilization_layer=stab,
    )
    return terminal, built, trace


def trace_record(*, query_id: str, trace: LayerTrace, stab_eps: float, stab_k: int) -> dict[str, Any]:
    n = trace.num_layers
    stab = trace.stabilization_layer
    depth_fraction = depth_fraction_list(n)
    return {
        "query_id": query_id,
        "num_layers": n,
        "depth_fraction": [round(v, 6) for v in depth_fraction],
        "margin": [round(v, 6) for v in trace.margins],
        "entropy": [round(v, 6) for v in trace.entropies],
        "slope_margin": round(trace.slope_margin, 6),
        "stabilization_layer": stab,
        "stabilization_frac": round(depth_fraction[stab - 1], 6) if n and 1 <= stab <= n else None,
        "stab_eps": stab_eps,
        "stab_k": stab_k,
    }


def append_trace(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def make_layerwise_row(
    *,
    query_id: str,
    row_uid: str | None,
    model_id: str,
    user_content: str,
    metrics: ProbeMetrics,
    built: dict,
    trace: LayerTrace,
) -> dict:
    row = make_probe_row(
        query_id=query_id,
        row_uid=row_uid,
        model_id=model_id,
        user_content=user_content,
        metrics=metrics,
        built=built,
    )
    row["extraction_method"] = PROBE_METHOD_LAYERWISE
    row["num_layers"] = trace.num_layers
    row["stabilization_layer"] = trace.stabilization_layer
    row["slope_margin"] = round(trace.slope_margin, 6)
    return row


def write_layerwise_row(path: Path, row: dict, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_LAYERWISE_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_layerwise_extraction(
    *,
    model: str,
    output: Path,
    trace_path: Path | None,
    dataset: str,
    split: str,
    limit: int,
    seed: int,
    device: str,
    dtype: str | None,
    stab_eps: float = STAB_EPS_DEFAULT,
    stab_k: int = STAB_K_DEFAULT,
    margin_tol: float = MARGIN_TOL_DEFAULT,
    overwrite: bool = False,
    batch_size: int = 1,
    query_filter: set[str] | None = None,
) -> int:
    if batch_size != 1:
        print(f"warning: layerwise uses batch_size=1 per query (got {batch_size}); ignoring for now")
    _prepare_output_path(output, overwrite=overwrite)
    if trace_path is not None:
        _prepare_output_path(trace_path, overwrite=overwrite)

    model_obj, tokenizer, device_res = load_model_and_tokenizer(model, device=device, dtype=dtype)
    print(f"device: {device_res}  layerwise  eps={stab_eps} k={stab_k} margin_tol={margin_tol}")

    write_header = True

    queries = load_queries(dataset, split, limit, seed)
    if query_filter is not None:
        queries = filter_queries(queries, query_filter)
        print(f"query_filter: {len(queries)} queries")

    for q in queries:
        terminal, built, trace = probe_layerwise(
            model_obj,
            tokenizer,
            q["user_content"],
            stab_eps=stab_eps,
            stab_k=stab_k,
            margin_tol=margin_tol,
        )
        row = make_layerwise_row(
            query_id=q["id"],
            row_uid=q.get("row_uid"),
            model_id=model,
            user_content=q["user_content"],
            metrics=terminal,
            built=built,
            trace=trace,
        )
        write_layerwise_row(output, row, write_header)
        write_header = False
        if trace_path is not None:
            append_trace(trace_path, trace_record(query_id=q["id"], trace=trace, stab_eps=stab_eps, stab_k=stab_k))
        print(
            f"{q['id']}: m={row['margin']:.4f} "
            f"stab={row['stabilization_layer']}/{trace.num_layers} "
            f"slope={row['slope_margin']:.4f}"
        )

    release_model(model_obj)
    print(f"Wrote {output}")
    if trace_path:
        print(f"Wrote {trace_path}")
    return 0
