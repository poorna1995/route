"""Layerwise prefill probes — margin trajectories (Route A) and repr drift (Route B).

Route A (RH5): logit-lens margin / entropy at each layer; ``stabilization_layer``,
``slope_margin``.

Route B (RH5-repr): adjacent hidden-state drift — ``total_representation_drift``,
``mean_adjacent_cos``, ``repr_adjacent_std``. No LM head; computed from the same
``hidden_states`` tuple as Route A.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    adjacent_cos: list[float]
    drift: list[float]
    total_representation_drift: float
    mean_adjacent_cos: float
    repr_adjacent_std: float


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


def drift_depth_fraction_list(num_layers: int) -> list[float]:
    """Normalized depth at layer ℓ for transition ℓ→ℓ+1; length L−1."""
    if num_layers <= 1:
        return []
    return [(i + 1) / num_layers for i in range(num_layers - 1)]


def compute_representation_drift(
    hidden_states: tuple,
    batch_idx: int,
    last_pos: int,
    n_layers: int,
) -> tuple[list[float], list[float], float, float, float]:
    """Route B — adjacent cos(h_ℓ, h_{ℓ+1}) and per-step drift 1−cos."""
    adjacent_cos: list[float] = []
    with torch.inference_mode():
        for i in range(n_layers - 1):
            h_a = hidden_states[i + 1][batch_idx, last_pos, :]
            h_b = hidden_states[i + 2][batch_idx, last_pos, :]
            cos = float(F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0), dim=1).item())
            adjacent_cos.append(cos)
    drift = [1.0 - c for c in adjacent_cos]
    total = float(sum(drift))
    mean_cos = float(sum(adjacent_cos) / len(adjacent_cos)) if adjacent_cos else 0.0
    std_drift = float(np.std(drift, ddof=0)) if drift else 0.0
    return adjacent_cos, drift, total, mean_cos, std_drift


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


def _softmax_margin(logits_row: torch.Tensor) -> float:
    p = torch.softmax(logits_row, dim=-1)
    k2 = min(2, int(p.shape[-1]))
    if k2 < 2:
        return 0.0
    top2 = torch.topk(p, k=k2).values
    return float(top2[0] - top2[1])


def _margin_delta_logits(logits_ref: torch.Tensor, h: torch.Tensor, model) -> float:
    with torch.inference_mode():
        logits_c = model.lm_head(h)
        return abs(_softmax_margin(logits_ref) - _softmax_margin(logits_c))


@torch.inference_mode()
def _terminal_hidden_vector(
    model,
    out,
    *,
    inputs: dict[str, torch.Tensor],
    batch_idx: int,
    last_pos: int,
) -> torch.Tensor:
    """Normed hidden at ``last_pos`` — same vector CausalLM feeds to ``lm_head``."""
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[batch_idx, last_pos]

    logits_ref = out.logits[batch_idx, last_pos]
    hs = out.hidden_states
    n_layers = int(model.config.num_hidden_layers)
    norm = _final_norm(model)

    candidates: list[torch.Tensor] = []
    if hs is not None:
        if len(hs) >= 1:
            # HF may append post-norm state as the final tuple entry.
            candidates.append(hs[-1][batch_idx, last_pos, :])
        if len(hs) >= n_layers + 1:
            candidates.append(norm(hs[n_layers][batch_idx, last_pos, :]))

    inner = getattr(model, "model", None)
    if inner is not None:
        base_out = inner(**inputs, output_hidden_states=False)
        base_lhs = getattr(base_out, "last_hidden_state", None)
        if base_lhs is None and isinstance(base_out, tuple) and base_out:
            base_lhs = base_out[0]
        if base_lhs is not None:
            candidates.append(base_lhs[batch_idx, last_pos])

    if not candidates:
        raise RuntimeError("cannot resolve terminal hidden state: no candidates")

    best_h = min(candidates, key=lambda h: _margin_delta_logits(logits_ref, h, model))
    best_margin_delta = _margin_delta_logits(logits_ref, best_h, model)
    if best_margin_delta > MARGIN_TOL_DEFAULT:
        raise RuntimeError(
            f"cannot resolve terminal hidden state: best margin Δ={best_margin_delta:.6f}"
        )
    return best_h


def verify_terminal_logit_parity(
    model,
    tokenizer,
    user_content: str,
    *,
    query_id: str = "smoke",
    margin_tol: float = MARGIN_TOL_DEFAULT,
) -> TerminalParityResult:
    """Smoke: terminal lm_head(last_hidden_state) must match out.logits at last prompt token."""
    built = build_chat_prompt(tokenizer, user_content)
    device = resolve_input_device(model)
    inputs = tokenizer(built["chat_prompt"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)

        batch_idx = 0
        last_pos = int(inputs["attention_mask"].sum(dim=1).item() - 1)
        logits_ref = out.logits[batch_idx, last_pos]
        h_final = _terminal_hidden_vector(
            model, out, inputs=inputs, batch_idx=batch_idx, last_pos=last_pos
        )
        logits_lens = model.lm_head(h_final)

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
    print("PASS: terminal lm_head(last_hidden_state) matches out.logits")
    return 0


def _layer_logits(
    model,
    hidden_states: tuple,
    batch_idx: int,
    last_pos: int,
    layer_idx: int,
    n_layers: int,
) -> torch.Tensor:
    with torch.inference_mode():
        h = hidden_states[layer_idx + 1][batch_idx, last_pos, :]
        if layer_idx == n_layers - 1:
            h = _final_norm(model)(h)
        return model.lm_head(h)


def probe_layerwise(
    model,
    tokenizer,
    user_content: str,
    *,
    stab_eps: float = STAB_EPS_DEFAULT,
    stab_k: int = STAB_K_DEFAULT,
    margin_tol: float = MARGIN_TOL_DEFAULT,
    repr_only: bool = False,
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
            raise RuntimeError(
                f"hidden_states missing layers (got {len(hs) if hs else 0}, need {n_layers + 1})"
            )

        adjacent_cos, drift, total_drift, mean_adj_cos, std_drift = compute_representation_drift(
            hs, batch_idx, last_pos, n_layers
        )

        margins: list[float] = []
        entropies: list[float] = []
        if repr_only:
            # Intermediate margins omitted (no LM-head pass); NaN keeps F7 from fake zeros.
            margins = [float("nan")] * (n_layers - 1) + [terminal.margin]
            entropies = [float("nan")] * (n_layers - 1) + [terminal.entropy]
        else:
            for i in range(n_layers - 1):
                metrics = extract_logits(
                    _layer_logits(model, hs, batch_idx, last_pos, i, n_layers),
                    tokenizer,
                )
                margins.append(metrics.margin)
                entropies.append(metrics.entropy)
            margins.append(terminal.margin)
            entropies.append(terminal.entropy)

        slope, stab = formation_scalars(margins, stab_eps, stab_k)
        trace = LayerTrace(
            num_layers=n_layers,
            margins=margins,
            entropies=entropies,
            slope_margin=slope,
            stabilization_layer=stab,
            adjacent_cos=adjacent_cos,
            drift=drift,
            total_representation_drift=total_drift,
            mean_adjacent_cos=mean_adj_cos,
            repr_adjacent_std=std_drift,
        )
    return terminal, built, trace


def trace_record(*, query_id: str, trace: LayerTrace, stab_eps: float, stab_k: int) -> dict[str, Any]:
    n = trace.num_layers
    stab = trace.stabilization_layer
    depth_fraction = depth_fraction_list(n)
    drift_depth = drift_depth_fraction_list(n)

    def _round_seq(values: list[float]) -> list[float | None]:
        out: list[float | None] = []
        for v in values:
            if v != v:  # NaN
                out.append(None)
            else:
                out.append(round(v, 6))
        return out

    return {
        "query_id": query_id,
        "num_layers": n,
        "depth_fraction": [round(v, 6) for v in depth_fraction],
        "drift_depth_fraction": [round(v, 6) for v in drift_depth],
        "margin": _round_seq(trace.margins),
        "entropy": _round_seq(trace.entropies),
        "slope_margin": round(trace.slope_margin, 6),
        "stabilization_layer": stab,
        "stabilization_frac": round(depth_fraction[stab - 1], 6) if n and 1 <= stab <= n else None,
        "adjacent_cos": [round(v, 6) for v in trace.adjacent_cos],
        "drift": [round(v, 6) for v in trace.drift],
        "total_representation_drift": round(trace.total_representation_drift, 6),
        "mean_adjacent_cos": round(trace.mean_adjacent_cos, 6),
        "repr_adjacent_std": round(trace.repr_adjacent_std, 6),
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
    row["total_representation_drift"] = round(trace.total_representation_drift, 6)
    row["mean_adjacent_cos"] = round(trace.mean_adjacent_cos, 6)
    row["repr_adjacent_std"] = round(trace.repr_adjacent_std, 6)
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
    repr_only: bool = False,
) -> int:
    if batch_size != 1:
        print(f"warning: layerwise uses batch_size=1 per query (got {batch_size}); ignoring for now")
    _prepare_output_path(output, overwrite=overwrite)
    if trace_path is not None:
        _prepare_output_path(trace_path, overwrite=overwrite)

    model_obj, tokenizer, device_res = load_model_and_tokenizer(model, device=device, dtype=dtype)
    mode = "repr-only" if repr_only else "margin+repr"
    print(f"device: {device_res}  layerwise ({mode})  eps={stab_eps} k={stab_k} margin_tol={margin_tol}")

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
            repr_only=repr_only,
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
            f"drift={row['total_representation_drift']:.4f} "
            f"mean_cos={row['mean_adjacent_cos']:.4f} "
            f"stab={row['stabilization_layer']}/{trace.num_layers}"
        )

    release_model(model_obj)
    print(f"Wrote {output}")
    if trace_path:
        print(f"Wrote {trace_path}")
    return 0
