"""Model-dependent prefill probes: entropy H and margin m (Study II / RH2)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from routing.constants import PROBE_CSV_FIELDS
from routing.datasets import load_queries
from routing.model_utils import (
    load_model_and_tokenizer,
    release_model,
    resolve_input_device,
)
from routing.prompt_protocol import build_chat_prompt

# Headline model-dependent signals: entropy, margin (Study II).
# Other columns are exploratory / audit only.

@dataclass(frozen=True)
class ProbeMetrics:
    entropy: float
    entropy_norm: float
    entropy_top10: float
    n_eff: float
    margin: float
    max_prob: float
    top5_mass: float
    top1_token: str
    top2_token: str
    vocab_size: int


def extract_logits(logits_row: torch.Tensor, tokenizer) -> ProbeMetrics:
    """Statistics at probe step T (last prompt token → P(first answer token))."""
    vocab_size = int(logits_row.shape[-1])
    log_p = F.log_softmax(logits_row, dim=-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum().item()
    max_entropy = math.log(vocab_size) if vocab_size > 1 else 0.0
    entropy_norm = (entropy / max_entropy) if max_entropy > 0 else 0.0
    n_eff = math.exp(entropy)
    k10 = min(10, vocab_size)
    top10 = torch.topk(p, k=k10)
    entropy_top10 = -(top10.values * top10.values.log()).sum().item()
    k2 = min(2, vocab_size)
    top2 = torch.topk(p, k=k2)
    margin = (top2.values[0] - top2.values[1]).item() if k2 > 1 else 0.0
    max_prob = top2.values[0].item()
    k5 = min(5, vocab_size)
    top5_mass = torch.topk(p, k=k5).values.sum().item()
    top1_id = int(top2.indices[0].item())
    top2_id = int(top2.indices[1].item()) if k2 > 1 else top1_id
    top1_token = tokenizer.convert_ids_to_tokens([top1_id])[0]
    top2_token = tokenizer.convert_ids_to_tokens([top2_id])[0]
    return ProbeMetrics(
        entropy=entropy,
        entropy_norm=entropy_norm,
        entropy_top10=entropy_top10,
        n_eff=n_eff,
        margin=margin,
        max_prob=max_prob,
        top5_mass=top5_mass,
        top1_token=top1_token,
        top2_token=top2_token,
        vocab_size=vocab_size,
    )


def first_token_logits(model, inputs: dict) -> torch.Tensor:
    out = model(**inputs)
    return out.logits[0, -1, :]


def probe_one(model, tokenizer, user_content: str) -> tuple[ProbeMetrics, dict[str, Any]]:
    built = build_chat_prompt(tokenizer, user_content)
    device = resolve_input_device(model)
    inputs = tokenizer(built["chat_prompt"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = first_token_logits(model, inputs)
    return extract_logits(logits, tokenizer), built


def probe_batch(
    model, tokenizer, user_contents: list[str]
) -> list[tuple[ProbeMetrics, dict[str, Any]]]:
    if len(user_contents) == 1:
        return [probe_one(model, tokenizer, user_contents[0])]

    built_list = [build_chat_prompt(tokenizer, uc) for uc in user_contents]
    prompts = [b["chat_prompt"] for b in built_list]
    device = resolve_input_device(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    last_idx = inputs["attention_mask"].sum(dim=1) - 1
    return [
        (extract_logits(logits[i, last_idx[i].item(), :], tokenizer), built)
        for i, built in enumerate(built_list)
    ]


def make_probe_row(
    *,
    query_id: str,
    row_uid: str | None,
    model_id: str,
    user_content: str,
    metrics: ProbeMetrics,
    built: dict,
) -> dict:
    return {
        "query_id": query_id,
        "row_uid": row_uid,
        "model_id": model_id,
        "user_content": user_content,
        "entropy": round(metrics.entropy, 6),
        "entropy_norm": round(metrics.entropy_norm, 6),
        "entropy_top10": round(metrics.entropy_top10, 6),
        "n_eff": round(metrics.n_eff, 6),
        "margin": round(metrics.margin, 6),
        "max_prob": round(metrics.max_prob, 6),
        "top5_mass": round(metrics.top5_mass, 6),
        "top1_token": metrics.top1_token,
        "top2_token": metrics.top2_token,
        "vocab_size": metrics.vocab_size,
        "prompt_tokens": built["prompt_tokens"],
        "protocol_version": built["protocol_version"],
        "prompt_hash": built["prompt_hash"],
        "chat_template": built["chat_template"],
        "tokenizer_id": built["tokenizer_id"],
        "extraction_method": "prefill_probe",
    }


def write_probe_row(path: Path, row: dict, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_probe_extraction(
    *,
    model: str,
    output: Path,
    prompt: str | None,
    query_id: str,
    dataset: str | None,
    split: str,
    limit: int,
    seed: int,
    device: str,
    dtype: str | None,
    batch_size: int,
    query_filter: set[str] | None = None,
) -> int:
    model_obj, tokenizer, device_res = load_model_and_tokenizer(model, device=device, dtype=dtype)
    print(f"device: {device_res}  batch_size: {batch_size}")

    write_header = not output.exists()

    if dataset:
        queries = load_queries(dataset, split, limit, seed)
        if query_filter is not None:
            from routing.datasets import filter_queries
            queries = filter_queries(queries, query_filter)
            print(f"query_filter: {len(queries)} queries")
        for start in range(0, len(queries), batch_size):
            chunk = queries[start : start + batch_size]
            probes = probe_batch(model_obj, tokenizer, [q["user_content"] for q in chunk])
            for q, (metrics, built) in zip(chunk, probes):
                row = make_probe_row(
                    query_id=q["id"],
                    row_uid=q.get("row_uid"),
                    model_id=model,
                    user_content=q["user_content"],
                    metrics=metrics,
                    built=built,
                )
                write_probe_row(output, row, write_header)
                write_header = False
                print(
                    f"{q['id']}: H={row['entropy']:.4f} m={row['margin']:.4f} "
                    f"top1={row['top1_token']!r} top2={row['top2_token']!r} "
                    f"hash={row['prompt_hash'][:8]}"
                )
    else:
        metrics, built = probe_one(model_obj, tokenizer, prompt or "")
        row = make_probe_row(
            query_id=query_id,
            row_uid=query_id,
            model_id=model,
            user_content=prompt or "",
            metrics=metrics,
            built=built,
        )
        write_probe_row(output, row, write_header)
        print(f"query_id: {row['query_id']}  top1={row['top1_token']!r} top2={row['top2_token']!r}")

    release_model(model_obj)
    print(f"Wrote {output}")
    return 0
