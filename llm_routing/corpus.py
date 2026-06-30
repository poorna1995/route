"""Load evaluation corpus C; partition H | R_c | R_t; oracle results."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset


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
            query_id=row.get("query_id") or row["example_id"],
            dataset=row.get("dataset") or row["benchmark"],
            text=row.get("text") or row["question"],
            choices=tuple(row["choices"]),
            answer_index=row["answer_index"],
            metadata=row.get("metadata", {}),
        )


@dataclass(frozen=True)
class QueryResult:
    """Oracle output for one (query, model) pair — signals and cost attach here."""

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
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: row[k] for k in fields if k in row})


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    config: str | None
    corpus_splits: tuple[str, ...]
    exclude_splits: tuple[str, ...] = ()


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "ARC-Challenge": DatasetSpec(
        "ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge",
        ("validation", "test"), ("train",),
    ),
    "MMLU_PRO": DatasetSpec(
        "MMLU_PRO", "TIGER-Lab/MMLU-Pro", "default",
        ("validation", "test"), ("dev", "auxiliary_train"),
    ),
}

_DATASET_ALIASES = {"TruthfulQA MCQ": "TruthfulQA-MC"}


def get_dataset(name: str) -> DatasetSpec:
    key = _DATASET_ALIASES.get(name, name)
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}")
    return DATASET_REGISTRY[key]


def _query_id(dataset: str, source_split: str, row_index: int) -> str:
    return f"{dataset}:{source_split}:{row_index}"


def _parse_arc(row: dict[str, Any], spec: DatasetSpec, source_split: str, row_index: int) -> Query:
    labels = list(row["choices"]["label"])
    key = row["answerKey"].strip().upper()
    if key in labels:
        answer_index = labels.index(key)
    elif len(key) == 1 and key.isalpha():
        answer_index = ord(key) - ord("A")
    else:
        raise ValueError(f"Bad answerKey {row['answerKey']!r}")
    choices = tuple(row["choices"]["text"])
    return Query(
        _query_id(spec.name, source_split, row_index), spec.name,
        row["question"].strip(), choices, answer_index,
        {"native_id": row["id"]},
    )


def _parse_mmlu_pro(row: dict[str, Any], spec: DatasetSpec, source_split: str, row_index: int) -> Query:
    choices = row.get("options") or row.get("choices")
    if not choices:
        raise ValueError("MMLU-Pro row missing options/choices")
    if row.get("answer_index") is not None:
        answer_index = int(row["answer_index"])
    else:
        key = str(row["answer"]).strip().upper()
        if len(key) == 1 and key.isalpha():
            answer_index = ord(key) - ord("A")
        else:
            raise ValueError(f"Bad MMLU-Pro answer {row.get('answer')!r}")
    subject = row.get("subject") or row.get("category", "unknown")
    return Query(
        _query_id(spec.name, source_split, row_index), spec.name,
        row["question"].strip(), tuple(str(c) for c in choices), answer_index,
        {"subject": subject},
    )


def _parse_truthfulqa(row: dict[str, Any], spec: DatasetSpec, source_split: str, row_index: int) -> Query:
    mc1 = row["mc1_targets"]
    labels = [int(x) for x in mc1["labels"]]
    if labels.count(1) != 1:
        raise ValueError("TruthfulQA MC1: need exactly one correct label")
    return Query(
        _query_id(spec.name, source_split, row_index), spec.name,
        row["question"].strip(), tuple(str(c) for c in mc1["choices"]), labels.index(1),
    )


def _parse_hellaswag(row: dict[str, Any], spec: DatasetSpec, source_split: str, row_index: int) -> Query:
    text = row.get("ctx") or f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
    return Query(
        _query_id(spec.name, source_split, row_index), spec.name,
        text, tuple(str(c) for c in row["endings"]), int(row["label"]),
    )


_ROW_PARSERS: dict[str, Callable[..., Query]] = {
    "ARC-Challenge": _parse_arc,
    "MMLU_PRO": _parse_mmlu_pro,
}


def load_corpus(spec: DatasetSpec, *, cache_dir: str | None = None) -> list[Query]:
    corpus: list[Query] = []
    parse = _ROW_PARSERS[spec.name]
    for source_split in spec.corpus_splits:
        if source_split in spec.exclude_splits:
            raise ValueError(f"Split {source_split!r} excluded for {spec.name}")
        load_kwargs: dict[str, Any] = {"path": spec.repo, "split": source_split}
        if spec.config:
            load_kwargs["name"] = spec.config
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        for row_index, row in enumerate(load_dataset(**load_kwargs)):
            query = parse(row, spec, source_split, row_index)
            if len(query.choices) < 2 or not 0 <= query.answer_index < len(query.choices):
                raise ValueError(f"Invalid query {query.query_id}")
            corpus.append(query)
    if len({q.query_id for q in corpus}) != len(corpus):
        raise ValueError(f"Duplicate query_id in {spec.name}")
    return corpus


def _normalize_query(q: Query) -> Query:
    return Query(
        q.query_id,
        q.dataset,
        q.text.strip(),
        tuple(c.strip() for c in q.choices),
        int(q.answer_index),
        dict(q.metadata),
    )


def _valid_query(q: Query) -> bool:
    return bool(q.text.strip()) and len(q.choices) >= 2 and 0 <= q.answer_index < len(q.choices)


def _stratum_key(q: Query, metadata_key: str | None) -> str:
    if metadata_key:
        return str(q.metadata.get(metadata_key, "unknown"))
    return q.dataset


def _stratified_subsample(
    corpus: list[Query], target_n: int, *, metadata_key: str | None, seed: int
) -> list[Query]:
    if target_n >= len(corpus):
        return list(corpus)
    rng = random.Random(seed)
    groups: dict[str, list[Query]] = {}
    for q in corpus:
        groups.setdefault(_stratum_key(q, metadata_key), []).append(q)
    for items in groups.values():
        rng.shuffle(items)

    n = len(corpus)
    keys = sorted(groups)
    counts = {k: int(target_n * len(groups[k]) / n) for k in keys}
    remainder = target_n - sum(counts.values())
    if remainder:
        ranked = sorted(
            keys,
            key=lambda k: (target_n * len(groups[k]) / n - counts[k]),
            reverse=True,
        )
        for k in ranked[:remainder]:
            counts[k] += 1

    out: list[Query] = []
    for k in keys:
        out.extend(groups[k][: counts[k]])
    rng.shuffle(out)
    return out[:target_n]


def prepare_corpus(
    corpus: list[Query],
    cfg: dict[str, Any] | None,
    *,
    seed: int = 42,
) -> tuple[list[Query], dict[str, Any]]:
    """M1 dataset preparation: normalize, drop invalid rows, optional stratified subsample."""
    cfg = cfg or {}
    meta: dict[str, Any] = {"n_loaded": len(corpus), "n_dropped_invalid": 0, "subsampled": False}

    cleaned: list[Query] = []
    for q in corpus:
        try:
            nq = _normalize_query(q)
        except (TypeError, ValueError):
            meta["n_dropped_invalid"] += 1
            continue
        if not _valid_query(nq):
            meta["n_dropped_invalid"] += 1
            continue
        cleaned.append(nq)

    meta["n_after_clean"] = len(cleaned)
    threshold = cfg.get("subsample_when_above")
    target = cfg.get("target_size")
    if threshold is not None and target is not None and len(cleaned) > int(threshold):
        cleaned = _stratified_subsample(
            cleaned,
            int(target),
            metadata_key=cfg.get("stratify_metadata_key"),
            seed=int(cfg.get("seed", seed)),
        )
        meta["subsampled"] = True
        meta["subsample_target"] = int(target)
        meta["subsample_threshold"] = int(threshold)

    meta["n_final"] = len(cleaned)
    if len({q.query_id for q in cleaned}) != len(cleaned):
        raise ValueError("duplicate query_id after corpus preparation")
    return cleaned, meta


def resolve_holdout_size(corpus_size: int, partition_cfg: dict[str, Any]) -> int:
    """M1/M2: fixed selection holdout count (same pilot cost on every benchmark)."""
    if "selection_holdout_n" in partition_cfg:
        sel_n = int(partition_cfg["selection_holdout_n"])
    elif pilot := partition_cfg.get("pilot"):
        sel_n = int(pilot.get("selection_holdout_n", pilot.get("selection_n", 150)))
    else:
        raise KeyError("partition.selection_holdout_n required")
    if sel_n >= corpus_size:
        raise ValueError(f"|C|={corpus_size} too small for selection_holdout_n={sel_n}")
    return sel_n


def resolve_test_size(partition_cfg: dict[str, Any], *, eval_pool_n: int) -> int:
    """M3: |R_t| from |C\\H|. Frozen test_n in setting overrides the policy rule."""
    if "test_n" in partition_cfg:
        return int(partition_cfg["test_n"])
    if "test_fraction" in partition_cfg:
        frac = float(partition_cfg["test_fraction"])
        tmin = int(partition_cfg.get("test_min", 150))
        tmax = int(partition_cfg.get("test_max", 1000))
        raw = round(frac * eval_pool_n)
        test_n = max(tmin, min(tmax, raw))
        if test_n >= eval_pool_n:
            raise ValueError(
                f"|C\\H|={eval_pool_n} too small for test rule "
                f"(fraction={frac}, min={tmin}, max={tmax} → {test_n})"
            )
        return test_n
    block = partition_cfg.get("eval") or partition_cfg.get("test")
    if isinstance(block, dict) and "test_n" in block:
        return int(block["test_n"])
    raise KeyError(
        "partition.test_fraction (+ test_min/test_max) or partition.test_n required for M3"
    )


PARTITION_METHOD_SPLIT_DATASET = "split_dataset"
_PARTITION_METHOD_ALIASES = {"random_split": PARTITION_METHOD_SPLIT_DATASET}


def _normalize_partition_method(method: str) -> str:
    return _PARTITION_METHOD_ALIASES.get(method, method)


def _shuffled_ids(corpus: list[Query], seed: int) -> list[str]:
    perm = list(range(len(corpus)))
    random.Random(seed).shuffle(perm)
    return [corpus[i].query_id for i in perm]


def partition_holdout(
    corpus: list[Query],
    *,
    method: str,
    seed: int,
    selection_n: int,
) -> dict[str, list[str]]:
    """M1: sample H ⊂ C only. Calib/test are created at M3 on the winning benchmark."""
    method = _normalize_partition_method(method)
    if method != PARTITION_METHOD_SPLIT_DATASET:
        raise NotImplementedError(f"partition method {method!r} not implemented")
    ids = _shuffled_ids(corpus, seed)
    return {"selection_holdout": ids[:selection_n]}


def partition_eval(
    corpus: list[Query],
    holdout_ids: list[str],
    *,
    method: str,
    seed: int,
    test_n: int,
) -> dict[str, list[str]]:
    """M3: split C \\ H into R_c and R_t using the same shuffle order as M1."""
    method = _normalize_partition_method(method)
    if method != PARTITION_METHOD_SPLIT_DATASET:
        raise NotImplementedError(f"partition method {method!r} not implemented")
    ids = _shuffled_ids(corpus, seed)
    expected = ids[: len(holdout_ids)]
    if holdout_ids != expected:
        raise ValueError("holdout IDs do not match partition seed — cannot extend to eval splits")
    rest = ids[len(holdout_ids) :]
    if test_n >= len(rest):
        raise ValueError(f"|C\\H|={len(rest)} too small for eval.test_n={test_n}")
    return {
        "selection_holdout": holdout_ids,
        "test": rest[:test_n],
        "calib": rest[test_n:],
    }


def validate_holdout(corpus: list[Query], holdout_ids: list[str]) -> None:
    ids = {q.query_id for q in corpus}
    if len(holdout_ids) != len(set(holdout_ids)):
        raise ValueError("holdout overlap")
    missing = [qid for qid in holdout_ids if qid not in ids]
    if missing:
        raise ValueError(f"holdout IDs not in corpus, e.g. {missing[0]!r}")


def validate_partition(corpus: list[Query], partition: dict[str, list[str]]) -> None:
    """Full M3 partition: H, R_c, R_t disjoint and cover C."""
    ids = {q.query_id for q in corpus}
    for key in ("selection_holdout", "calib", "test"):
        if not partition.get(key):
            raise ValueError(f"partition.{key} missing — run M3 eval first")
    all_parts = partition["selection_holdout"] + partition["calib"] + partition["test"]
    if set(all_parts) != ids:
        raise ValueError("partition IDs do not match corpus")
    if len(set(all_parts)) != len(all_parts):
        raise ValueError("partition overlap")


def eval_query_ids(
    partition: dict[str, list[str]],
    *,
    query_ids: list[str] | None = None,
) -> tuple[list[str], set[str], set[str]]:
    """Return (R_c ∪ R_t) IDs in stable order; reject holdout leakage."""
    if not partition.get("calib") or not partition.get("test"):
        raise ValueError("M3 eval partition required — run: python run.py eval --run <run_dir>")

    holdout = set(partition["selection_holdout"])
    calib = set(partition["calib"])
    test = set(partition["test"])
    if holdout & calib or holdout & test or calib & test:
        raise ValueError("partition overlap between holdout, calib, and test")

    eval_ids = partition["calib"] + partition["test"]
    if holdout & set(eval_ids):
        raise ValueError("selection holdout leaked into eval query set")

    if query_ids is None:
        return eval_ids, calib, test

    extra = set(query_ids) - calib - test
    if extra:
        sample = next(iter(extra))
        if sample in holdout:
            raise ValueError(f"query_ids include selection holdout ID {sample!r}")
        raise ValueError(f"query_ids include unknown ID {sample!r}")
    return query_ids, calib, test


def save_corpus(
    out_dir: Path,
    corpus: list[Query],
    manifest: dict[str, Any],
    partition: dict[str, list[str]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for query in corpus:
            f.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    if partition:
        if partition.get("calib") and partition.get("test"):
            validate_partition(corpus, partition)
        else:
            validate_holdout(corpus, partition["selection_holdout"])
        with (out_dir / "partition.json").open("w", encoding="utf-8") as f:
            json.dump(partition, f, indent=2)
            f.write("\n")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL without str.splitlines() — U+0085 in field values must not split records."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus_artifacts(out_dir: Path) -> tuple[list[Query], dict[str, Any], dict[str, list[str]] | None]:
    corpus = [Query.from_dict(row) for row in _iter_jsonl(out_dir / "corpus.jsonl")]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    part_path = out_dir / "partition.json"
    partition = json.loads(part_path.read_text(encoding="utf-8")) if part_path.exists() else None
    if partition:
        if partition.get("calib") and partition.get("test"):
            validate_partition(corpus, partition)
        else:
            validate_holdout(corpus, partition["selection_holdout"])
    return corpus, manifest, partition


def select_queries(corpus: list[Query], query_ids: list[str]) -> list[Query]:
    by_id = {q.query_id: q for q in corpus}
    missing = [qid for qid in query_ids if qid not in by_id]
    if missing:
        raise KeyError(f"Missing {len(missing)} query IDs, e.g. {missing[0]!r}")
    return [by_id[qid] for qid in query_ids]


def write_jsonl(path: Path, rows: list[Any], to_dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, from_dict) -> list:
    if not path.exists():
        return []
    return [from_dict(row) for row in _iter_jsonl(path)]


def select_split(run, split: str, limit: int | None) -> list[Query]:
    """Select queries for a run split (H, R_c, or R_t)."""
    from llm_routing.paths import SPLITS

    corpus, _, partition = load_corpus_artifacts(run.corpus_dir)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split != "selection_holdout" and not (partition.get("calib") and partition.get("test")):
        raise ValueError(
            f"split={split!r} requires M3 eval partition — run: python run.py eval --run {run.root}"
        )
    queries = select_queries(corpus, partition[split])
    return queries[:limit] if limit else queries
