#!/usr/bin/env python3
"""One-query walkthrough: oracle → φ → ψ → χ (mock, no GPU, no datasets)."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Minimal corpus stub so we avoid `datasets` / HF on import.
@dataclass(frozen=True)
class Query:
    query_id: str
    dataset: str
    text: str
    choices: tuple[str, ...]
    answer_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Query:
        return cls(
            query_id=row["query_id"],
            dataset=row["dataset"],
            text=row["text"],
            choices=tuple(row["choices"]),
            answer_index=row["answer_index"],
            metadata=row.get("metadata", {}),
        )


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    model: str
    raw_output: str
    parsed_answer: int | None
    correct: int
    latency_ms: float | None = None
    token_count: int | None = None
    model_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> QueryResult:
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: row[k] for k in fields if k in row})


def write_jsonl(path: Path, records: list, to_dict: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(to_dict(r)) for r in records) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path, from_dict: Callable) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(from_dict(json.loads(line)))
    return out


corpus_stub = types.ModuleType("llm_routing.corpus")
corpus_stub.Query = Query
corpus_stub.QueryResult = QueryResult
corpus_stub.write_jsonl = write_jsonl
corpus_stub.read_jsonl = read_jsonl


def load_corpus_artifacts(out_dir: Path) -> tuple[list[Query], dict[str, Any], None]:
    corpus = [Query.from_dict(json.loads(line)) for line in (out_dir / "corpus.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return corpus, manifest, None


corpus_stub.load_corpus_artifacts = load_corpus_artifacts
sys.modules["llm_routing.corpus"] = corpus_stub

SETTING = {
    "pool": {
        "M_lo": "meta-llama/Llama-3.2-3B-Instruct",
        "M_hi": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "protocol": {
        "protocol_version": "mcq_letter",
        "system_prompt": "You answer multiple-choice questions.\n",
        "user_template": "{question}\n\n{choices}\n\nAnswer:\n",
        "decoding": {"temperature": 0.0, "max_tokens": 16},
    },
}

CONFIG = {
    "structural": {"mattr_window": 25, "zlib_level": 6},
    "embedding_geometry": {"pca_components": 3, "knn_k": 10},
}


def routing_bucket_name(y_lo: int, y_hi: int) -> str:
    if y_lo and y_hi:
        return "easy"
    if not y_lo and y_hi:
        return "opportunity"
    if y_lo and not y_hi:
        return "lo_only"
    return "too_hard"


def routing_oracle_r(y_lo: int, y_hi: int) -> int:
    return int(y_lo == 0 and y_hi == 1)

import contextlib

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda _text: {}
sys.modules["yaml"] = yaml_stub

torch_stub = types.ModuleType("torch")

@contextlib.contextmanager
def _inference_mode():
    yield

torch_stub.inference_mode = _inference_mode
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch_stub

from llm_routing.signals.chi.stage import extract_cross_model_signals  # noqa: E402
from llm_routing.signals.psi import (  # noqa: E402
    extract_model_response_signals,
    mock_protocol_trace,
)
from llm_routing.signals.psi.protocol import McqLetterProtocolExtractor  # noqa: E402
from llm_routing.oracle import run_oracle_inference_mock  # noqa: E402
from llm_routing.signals.phi.core import (  # noqa: E402
    QueryDerivedRecord,
    TokenCounter,
    canonical_user,
    extract_ambiguity,
    extract_structural,
)


def _pp(title: str, obj: object) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    protocol = SETTING["protocol"]
    config = CONFIG

    query = Query(
        query_id="ARC-Challenge:demo:0",
        dataset="ARC-Challenge",
        text="Which planet is closest to the Sun?",
        choices=("Earth", "Venus", "Mercury", "Mars"),
        answer_index=2,
    )

    print("INPUT QUERY")
    print(f"  id:      {query.query_id}")
    print(f"  text:    {query.text}")
    print(f"  choices: {query.choices}")
    print(f"  answer:  {chr(ord('A') + query.answer_index)} (index {query.answer_index})")

    lo_id = SETTING["pool"]["M_lo"]
    hi_id = SETTING["pool"]["M_hi"]
    lo_row = run_oracle_inference_mock(lo_id, [query], protocol)[0]
    hi_row = run_oracle_inference_mock(hi_id, [query], protocol)[0]

    _pp(
        "STAGE 2 — Oracle (M_lo): answer + model_response trace (GPU once, frozen)",
        {
            "query_id": lo_row.query_id,
            "model": lo_row.model,
            "raw_output": lo_row.raw_output,
            "parsed_answer": lo_row.parsed_answer,
            "correct": lo_row.correct,
            "model_response": lo_row.model_response,
        },
    )

    canonical = canonical_user(query, protocol)
    structural = extract_structural(query, canonical, TokenCounter(None), config)
    ambiguity = extract_ambiguity(query)
    # geometry normally fits on R_c embeddings (needs sklearn); mock for local demo
    embedding_geometry = {
        "pc1": 0.12,
        "pc2": -0.05,
        "pc3": 0.03,
        "centroid_distance": 0.18,
        "mean_knn_similarity": 0.78,
        "lof_score": -0.4,
    }

    phi = QueryDerivedRecord(
        query_id=query.query_id,
        split="demo",
        structural=structural,
        ambiguity=ambiguity,
        embedding_geometry=embedding_geometry,
    ).to_dict()
    _pp("STAGE 5A — φ(q) query-derived [H1, model-independent]", phi)

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        oracle_dir = run_root / "oracle"
        oracle_dir.mkdir()
        write_jsonl(oracle_dir / "M_lo.jsonl", [lo_row], QueryResult.to_dict)
        write_jsonl(oracle_dir / "M_hi.jsonl", [hi_row], QueryResult.to_dict)

        corpus_dir = run_root / "corpus"
        corpus_dir.mkdir()
        write_jsonl(corpus_dir / "corpus.jsonl", [query], Query.to_dict)
        (corpus_dir / "manifest.json").write_text(
            json.dumps({"dataset": query.dataset}) + "\n",
            encoding="utf-8",
        )

        psi_lo = json.loads(
            extract_model_response_signals(run_root, pool_role="M_lo").read_text().strip()
        )
        psi_hi = json.loads(
            extract_model_response_signals(run_root, pool_role="M_hi").read_text().strip()
        )
        chi = json.loads(extract_cross_model_signals(run_root).read_text().strip())

        _pp("STAGE 5B — ψ(q, M_lo) [H2, model-dependent]", psi_lo)
        _pp("STAGE 5B — ψ(q, M_hi) [H2, model-dependent]", psi_hi)
        _pp("STAGE 5C — χ(q) [H3, cross-model join]", chi)

    print(f"\n{'=' * 60}\nORACLE LABEL r(q) (Stage 6+)\n{'=' * 60}")
    y_lo, y_hi = lo_row.correct, hi_row.correct
    bucket = routing_bucket_name(y_lo, y_hi)
    r_q = routing_oracle_r(y_lo, y_hi)
    print(f"  M_lo: correct={y_lo}  letter={chr(ord('A') + lo_row.parsed_answer)}")
    print(f"  M_hi: correct={y_hi}  letter={chr(ord('A') + hi_row.parsed_answer)}")
    print(f"  Bucket: {bucket}")
    print(f"  r(q) = {r_q}   (1 only when M_lo wrong AND M_hi right → opportunity)")
    if bucket == "lo_only":
        print("  → No routing need. M_lo already solved it; escalating hurts accuracy.")
    elif bucket == "opportunity":
        print("  → M_hi was the appropriate model; routing to M_hi is valuable.")
    elif bucket == "easy":
        print("  → Both models succeed; prefer M_lo for cost.")
    else:
        print("  → too_hard: even M_hi fails; routing cannot help accuracy.")

    trace = mock_protocol_trace(protocol["protocol_version"], 4, 2, generated_text="C")
    _pp(
        "ψ metric meanings (example trace)",
        {
            **McqLetterProtocolExtractor().compute_metrics(trace),
            "_note": {
                "entropy": "spread over A/B/C/D (high = uncertain)",
                "msp": "max choice probability — routing signal, not calibrated P(correct)",
                "margin": "top1 - top2 probability",
                "predicted_logprob": "logprob of predicted letter",
                "mean_logprob": "deprecated alias for predicted_logprob",
            },
        },
    )


if __name__ == "__main__":
    main()
