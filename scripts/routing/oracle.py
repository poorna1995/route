"""Offline routing-opportunity assessment (C2) — full generation, bucket labels only."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from routing.datasets import load_queries
from routing.model_utils import (
    load_model_and_tokenizer,
    release_model,
    resolve_input_device,
)
from routing.prompt_protocol import build_chat_prompt


def extract_gsm8k_number(text: str) -> str | None:
    if "####" in text:
        tail = text.split("####")[-1].strip()
        nums = re.findall(r"-?\d+", tail.replace(",", ""))
        return nums[0] if nums else None
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def extract_arc_letter(text: str) -> str | None:
    upper = text.upper()
    matches = re.findall(r"\b([ABCD])\b", upper)
    return matches[-1] if matches else None


def extract_bool(text: str) -> str | None:
    lower = text.lower()
    if re.search(r"\bno\b", lower) and not re.search(r"\byes\b", lower):
        return "no"
    if re.search(r"\byes\b", lower) and not re.search(r"\bno\b", lower):
        return "yes"
    yes = re.findall(r"\b(yes|no)\b", lower)
    return yes[-1] if yes else None


def normalize(num: str | None) -> str | None:
    if num is None:
        return None
    try:
        return str(int(float(num)))
    except ValueError:
        return num.strip()


def score_response(query: dict, text: str) -> tuple[str | None, bool]:
    gold = query["gold"]
    eval_type = query["eval"]
    if eval_type == "numeric":
        pred = normalize(extract_gsm8k_number(text))
    elif eval_type == "letter":
        pred = extract_arc_letter(text)
    elif eval_type == "bool":
        pred = extract_bool(text)
        ok = pred is not None and pred == str(gold).lower()
        return pred, ok
    else:
        raise ValueError(f"Unknown eval type: {eval_type}")
    ok = gold is not None and pred is not None and pred == gold
    return pred, ok


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = resolve_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_model(
    model_id: str,
    queries: list[dict],
    max_new_tokens: int,
    *,
    device: str,
    dtype: str | None,
    done: dict[str, str] | None = None,
    on_progress: Callable[[str, str, int, str], None] | None = None,
    max_pending: int | None = None,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    if done is None:
        done = {}
    pending = [q for q in queries if q["id"] not in done]
    if max_pending is not None and max_pending > 0:
        pending = pending[:max_pending]
    if not pending:
        print(f"\n--- {model_id}: all {len(queries)} queries cached ---")
        return done, {}, {}

    print(f"\n--- loading {model_id} ({len(pending)} queries to run) ---")
    t0 = time.perf_counter()
    model, tokenizer, resolved = load_model_and_tokenizer(model_id, device=device, dtype=dtype)
    print(f"loaded on {resolved} in {time.perf_counter() - t0:.1f}s")

    prompt_tokens: dict[str, int] = {}
    prompt_hashes: dict[str, str] = {}
    for q in pending:
        built = build_chat_prompt(tokenizer, q["user_content"])
        prompt_tokens[q["id"]] = built["prompt_tokens"]
        prompt_hashes[q["id"]] = built["prompt_hash"]
        t1 = time.perf_counter()
        text = generate(model, tokenizer, built["chat_prompt"], max_new_tokens)
        done[q["id"]] = text
        print(
            f"  {q['id']}: {time.perf_counter() - t1:.1f}s "
            f"({built['prompt_tokens']} prompt tokens, hash={built['prompt_hash'][:8]})"
        )
        if on_progress:
            on_progress(q["id"], text, built["prompt_tokens"], built["prompt_hash"])

    release_model(model)
    return done, prompt_tokens, prompt_hashes


def classify_bucket(weak_ok: bool, strong_ok: bool) -> str:
    if weak_ok and strong_ok:
        return "easy"
    if not weak_ok and strong_ok:
        return "opportunity"
    if weak_ok and not strong_ok:
        return "weak_only"
    return "too_hard"


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def build_partial_rows(
    queries: list[dict],
    weak_out: dict[str, str],
    strong_out: dict[str, str],
    weak_tokens: dict[str, int],
    strong_tokens: dict[str, int],
    weak_hashes: dict[str, str],
    strong_hashes: dict[str, str],
) -> list[dict]:
    rows = []
    for q in queries:
        qid = q["id"]
        if qid not in weak_out and qid not in strong_out:
            continue
        row = {"id": qid, "row_uid": q.get("row_uid"), "user_content": q["user_content"], "gold": q["gold"]}
        if qid in weak_out:
            wp, wok = score_response(q, weak_out[qid])
            row.update({
                "weak_pred": wp,
                "weak_ok": wok,
                "weak_text": weak_out[qid],
                "prompt_tokens_weak": weak_tokens.get(qid),
                "prompt_hash_weak": weak_hashes.get(qid),
            })
        if qid in strong_out:
            sp, sok = score_response(q, strong_out[qid])
            row.update({
                "strong_pred": sp,
                "strong_ok": sok,
                "strong_text": strong_out[qid],
                "prompt_tokens_strong": strong_tokens.get(qid),
                "prompt_hash_strong": strong_hashes.get(qid),
            })
        if "weak_ok" in row and "strong_ok" in row:
            row["bucket"] = classify_bucket(row["weak_ok"], row["strong_ok"])
        rows.append(row)
    return rows


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_oracle_assessment(
    *,
    weak: str,
    strong: str,
    dataset: str,
    split: str,
    limit: int,
    seed: int,
    max_new_tokens: int,
    device: str,
    dtype: str | None,
    output: Path | None,
    strong_only: bool,
    weak_only: bool,
    no_resume: bool,
    query_filter: set[str] | None = None,
    max_pending: int | None = None,
) -> int:
    print(f"config: weak={weak}")
    print(f"        strong={strong}")
    print(f"        dataset={dataset}/{split} n={limit} seed={seed}")
    if query_filter:
        print(f"        query_filter: {len(query_filter)} ids")
    if max_pending:
        print(f"        max_pending: {max_pending} queries per model pass (resume batch)")
    print(f"        max_new_tokens={max_new_tokens} device={device}")

    queries = load_queries(dataset, split, limit, seed)
    if query_filter is not None:
        from routing.datasets import filter_queries
        queries = filter_queries(queries, query_filter)
    query_ids = {q["id"] for q in queries}

    weak_done: dict[str, str] = {}
    strong_done: dict[str, str] = {}
    weak_tokens: dict[str, int] = {}
    strong_tokens: dict[str, int] = {}
    weak_hashes: dict[str, str] = {}
    strong_hashes: dict[str, str] = {}

    if output and not no_resume:
        prev = load_checkpoint(output)
        if prev:
            for r in prev.get("rows", []):
                if r["id"] not in query_ids:
                    continue
                if r.get("weak_text"):
                    weak_done[r["id"]] = r["weak_text"]
                    if r.get("prompt_tokens_weak") is not None:
                        weak_tokens[r["id"]] = r["prompt_tokens_weak"]
                    if r.get("prompt_hash_weak"):
                        weak_hashes[r["id"]] = r["prompt_hash_weak"]
                if r.get("strong_text"):
                    strong_done[r["id"]] = r["strong_text"]
                    if r.get("prompt_tokens_strong") is not None:
                        strong_tokens[r["id"]] = r["prompt_tokens_strong"]
                    if r.get("prompt_hash_strong"):
                        strong_hashes[r["id"]] = r["prompt_hash_strong"]
            if weak_done or strong_done:
                print(f"resume: {len(weak_done)} weak, {len(strong_done)} strong from {output}")

    def flush_partial() -> None:
        if not output:
            return
        save_checkpoint(
            output,
            {
                "_partial": True,
                "weak_model": weak,
                "strong_model": strong,
                "dataset": f"{dataset}/{split}",
                "n": len(queries),
                "seed": seed,
                "rows": build_partial_rows(
                    queries, weak_done, strong_done,
                    weak_tokens, strong_tokens, weak_hashes, strong_hashes,
                ),
            },
        )

    def on_weak(qid: str, text: str, ptokens: int, phash: str) -> None:
        weak_tokens[qid] = ptokens
        weak_hashes[qid] = phash
        flush_partial()

    def on_strong(qid: str, text: str, ptokens: int, phash: str) -> None:
        strong_tokens[qid] = ptokens
        strong_hashes[qid] = phash
        flush_partial()

    if not strong_only:
        weak_out, wt, wh = run_model(
            weak, queries, max_new_tokens,
            device=device, dtype=dtype, done=weak_done, on_progress=on_weak,
            max_pending=max_pending,
        )
        weak_tokens.update(wt)
        weak_hashes.update(wh)
        flush_partial()
    else:
        weak_out = weak_done
        if len(weak_out) < len(queries):
            print(
                f"ERROR: --strong-only but only {len(weak_out)}/{len(queries)} "
                "weak results in checkpoint. Run weak pass first.",
                file=sys.stderr,
            )
            return 1

    if not weak_only:
        strong_out, st, sh = run_model(
            strong, queries, max_new_tokens,
            device=device, dtype=dtype, done=strong_done, on_progress=on_strong,
            max_pending=max_pending,
        )
        strong_tokens.update(st)
        strong_hashes.update(sh)
    else:
        strong_out = strong_done

    incomplete = [
        q for q in queries
        if q["id"] not in weak_out or q["id"] not in strong_out
    ]
    if incomplete:
        flush_partial()
        done_n = len(queries) - len(incomplete)
        print(
            f"\nPartial checkpoint: {done_n}/{len(queries)} queries have weak+strong. "
            f"Re-run the same command to continue (resume enabled)."
        )
        return 0

    rows = []
    counts = {"easy": 0, "opportunity": 0, "weak_only": 0, "too_hard": 0}

    for q in queries:
        qid = q["id"]
        gold = q["gold"]
        weak_pred, weak_ok = score_response(q, weak_out[qid])
        strong_pred, strong_ok = score_response(q, strong_out[qid])
        bucket = classify_bucket(weak_ok, strong_ok)
        counts[bucket] += 1
        rows.append({
            "id": qid,
            "row_uid": q.get("row_uid"),
            "user_content": q["user_content"],
            "gold": gold,
            "weak_pred": weak_pred,
            "strong_pred": strong_pred,
            "weak_ok": weak_ok,
            "strong_ok": strong_ok,
            "bucket": bucket,
            "weak_text": weak_out[qid],
            "strong_text": strong_out[qid],
            "prompt_tokens_weak": weak_tokens.get(qid),
            "prompt_tokens_strong": strong_tokens.get(qid),
            "prompt_hash_weak": weak_hashes.get(qid),
            "prompt_hash_strong": strong_hashes.get(qid),
        })

    n = len(rows)
    rates = {k: counts[k] / n for k in counts}
    summary = {
        "claim": "C2",
        "weak_model": weak,
        "strong_model": strong,
        "dataset": f"{dataset}/{split}",
        "n": n,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "counts": counts,
        "rates": rates,
        "rows": rows,
    }

    print("\n=== per-query ===")
    print(f"{'id':<16} {'weak':^6} {'strong':^6} {'bucket'}")
    for r in rows:
        w = "✔" if r["weak_ok"] else "✘"
        s = "✔" if r["strong_ok"] else "✘"
        print(f"{r['id']:<16} {w:^6} {s:^6} {r['bucket']}")

    print("\n=== aggregate (C2) ===")
    for k in ("opportunity", "weak_only", "easy", "too_hard"):
        print(f"  {k:12} {counts[k]:3d}  ({rates[k]*100:.1f}%)")

    if output:
        save_checkpoint(output, summary)
        print(f"\nWrote {output}")

    return 0
