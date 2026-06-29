"""Query-derived φ(q): config, per-query extraction, and calib-fitted engineering."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = ROOT / "experiments/query_derived_defaults.yaml"
JSONL_BLOCKS = ("structural", "ambiguity", "embedding_geometry")
MAX_PCA_COMPONENTS = 3

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)


# --- config / schema ---


@dataclass
class QueryDerivedRecord:
    query_id: str
    split: str
    structural: dict[str, Any]
    ambiguity: dict[str, Any]
    embedding_geometry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_record_blocks(record: dict[str, Any]) -> dict[str, Any]:
    """Map legacy jsonl block keys to structural / embedding_geometry."""
    rec = dict(record)
    if "load" in rec and "structural" not in rec:
        rec["structural"] = rec.pop("load")
    if "novelty" in rec and "embedding_geometry" not in rec:
        rec["embedding_geometry"] = rec.pop("novelty")
    if "geometry" in rec and "embedding_geometry" not in rec:
        rec["embedding_geometry"] = rec.pop("geometry")
    geo = rec.get("embedding_geometry")
    if isinstance(geo, dict) and "retrieval_density" in geo and "mean_knn_similarity" not in geo:
        geo = dict(geo)
        geo["mean_knn_similarity"] = geo.pop("retrieval_density")
        rec["embedding_geometry"] = geo
    return rec


def flatten_blocks(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten jsonl blocks to dotted keys; accepts legacy load/novelty block names."""
    rec = _normalize_record_blocks(record)
    out: dict[str, Any] = {}
    for block in JSONL_BLOCKS:
        for key, val in (rec.get(block) or {}).items():
            out[f"{block}.{key}"] = val
    return out


def load_query_derived_defaults(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULTS_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    return data


def structural_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("structural") or config.get("load") or config.get("lexical") or {}


def embedding_geometry_section(config: dict[str, Any]) -> dict[str, Any]:
    return (
        config.get("embedding_geometry")
        or config.get("geometry")
        or config.get("novelty")
        or {}
    )


# Deprecated aliases (old manifest / yaml keys).
load_section = structural_section
geometry_section = embedding_geometry_section
novelty_section = embedding_geometry_section


def resolve_tokenizer_id(setting: dict[str, Any], config: dict[str, Any]) -> str | None:
    source = config.get("tokenizer", {}).get("source", "pool_M_lo")
    if source == "pool_M_lo":
        return setting.get("pool", {}).get("M_lo")
    if source == "none":
        return None
    return str(source)


# --- φ_structural / φ_ambiguity extraction ---


def word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def windowed_type_token_ratio(tokens: list[str], window: int) -> float:
    n = len(tokens)
    if n == 0:
        return float("nan")
    if n < window:
        return len(set(tokens)) / n
    ratios: list[float] = []
    for start in range(n - window + 1):
        chunk = tokens[start : start + window]
        ratios.append(len(set(chunk)) / window)
    return statistics.fmean(ratios)


def compression_ratio(text: str, level: int) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return float("nan")
    return len(zlib.compress(raw, level=level)) / len(raw)


def word_jaccard(a: str, b: str) -> float:
    sa, sb = set(word_tokens(a)), set(word_tokens(b))
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


class TokenCounter:
    """HF tokenizer optional (count only, no forward pass)."""

    def __init__(self, tokenizer_id: str | None = None) -> None:
        self._tokenizer_id = tokenizer_id
        self._tokenizer: Any = None

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            if not self._tokenizer_id:
                raise RuntimeError("HF tokenizer requested but tokenizer_id is unset")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        return self._tokenizer

    def count(self, text: str) -> int:
        if self._tokenizer_id:
            return len(self._load_tokenizer().encode(text, add_special_tokens=False))
        return len(word_tokens(text))

    def chat_token_count(self, messages: list[dict[str, str]]) -> int:
        if self._tokenizer_id:
            tok = self._load_tokenizer()
            if getattr(tok, "chat_template", None):
                return len(
                    tok.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
            return sum(self.count(m["content"]) for m in messages)
        return sum(self.count(m["content"]) for m in messages)


def canonical_user(query: Any, protocol: dict[str, Any]) -> str:
    from llm_routing.oracle import render_user_message

    return render_user_message(query, protocol)


def extract_token_stats(
    query: Any,
    canonical: str,
    counter: TokenCounter,
) -> dict[str, float | int]:
    from llm_routing.oracle import format_question

    question = format_question(query)
    opt_lens = [counter.count(c) for c in query.choices]
    q_len = counter.count(question)
    opt_sum = sum(opt_lens) or 1
    return {
        "prompt_token_len": counter.count(canonical),
        "question_token_len": q_len,
        "option_count": len(query.choices),
        "mean_option_token_len": statistics.fmean(opt_lens) if opt_lens else 0.0,
        "std_option_token_len": statistics.pstdev(opt_lens) if len(opt_lens) > 1 else 0.0,
        "question_option_ratio": q_len / opt_sum,
    }


def extract_lexical(canonical: str, config: dict[str, Any]) -> dict[str, float]:
    lcfg = structural_section(config)
    tokens = word_tokens(canonical)
    window = int(lcfg.get("mattr_window", 50))
    level = int(lcfg.get("zlib_level", 6))
    return {
        "mattr": windowed_type_token_ratio(tokens, window),
        "compression_ratio": compression_ratio(canonical, level),
    }


def extract_structural(
    query: Any,
    canonical: str,
    counter: TokenCounter,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """φ_structural block: token-length stats + lexical density (8 scalars)."""
    out = extract_token_stats(query, canonical, counter)
    out.update(extract_lexical(canonical, config))
    return out


def extract_load(
    query: Any,
    canonical: str,
    counter: TokenCounter,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """Deprecated alias for extract_structural."""
    return extract_structural(query, canonical, counter, config)


def extract_ambiguity(query: Any) -> dict[str, float]:
    stem = query.text.strip()
    choices = [c.strip() for c in query.choices]
    stem_j = [word_jaccard(stem, c) for c in choices]
    pairs: list[float] = []
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            pairs.append(word_jaccard(choices[i], choices[j]))
    char_lens = [len(c) for c in choices]
    return {
        "stem_choice_overlap_max": max(stem_j) if stem_j else 0.0,
        "stem_choice_overlap_mean": statistics.fmean(stem_j) if stem_j else 0.0,
        "choice_choice_overlap": statistics.fmean(pairs) if pairs else 0.0,
        "choice_length_range": (max(char_lens) - min(char_lens)) if char_lens else 0.0,
    }


extract_mcq = extract_ambiguity


# --- φ_semantic / φ_embedding_geometry / z-score engineering ---


def _mock_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    while len(out) < dim:
        for i in range(0, len(digest), 4):
            chunk = digest[i : i + 4]
            if len(chunk) < 4:
                break
            out.append((int.from_bytes(chunk, "big") / 2**32) * 2.0 - 1.0)
            if len(out) >= dim:
                break
        digest = hashlib.sha256(digest).digest()
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


def encode_canonical_texts(
    texts: list[str],
    *,
    model_id: str,
    mock: bool = False,
    dimension: int = 384,
) -> list[list[float]]:
    if mock:
        return [_mock_embedding(t, dimension) for t in texts]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    vectors = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def save_embedding(path: Path, vector: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        np.save(path, np.asarray(vector, dtype=np.float32))
    except ImportError:
        path.write_text(json.dumps(vector) + "\n", encoding="utf-8")


@dataclass
class GeometryModel:
    """Embedding-geometry descriptors fit on R_c (PCA, kNN, LOF, centroid)."""

    pca_components: list[list[float]] | None = None
    centroid: list[float] | None = None
    knn_k: int = 10
    lof_offset: float = 0.0
    lof_scale: float = 1.0

    def fit(self, matrix: list[list[float]], config: dict[str, Any]) -> None:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

        gcfg = embedding_geometry_section(config)
        n_comp = min(int(gcfg.get("pca_components", MAX_PCA_COMPONENTS)), MAX_PCA_COMPONENTS)
        k = int(gcfg.get("knn_k", 10))

        x = np.asarray(matrix, dtype=np.float64)
        self.knn_k = k

        pca = PCA(n_components=min(n_comp, x.shape[0], x.shape[1]))
        pca.fit(x)
        self.pca_components = pca.components_.tolist()
        self.centroid = x.mean(axis=0).tolist()

        self._nn = NearestNeighbors(n_neighbors=min(k, len(matrix)), metric="cosine")
        self._nn.fit(x)

        lof = LocalOutlierFactor(n_neighbors=min(k, len(matrix)), novelty=True)
        lof.fit(x)
        self._lof = lof
        train_scores = -lof.score_samples(x)
        self.lof_offset = float(train_scores.mean())
        self.lof_scale = float(train_scores.std()) or 1.0

    def transform(self, vector: list[float]) -> dict[str, float]:
        import numpy as np

        if self.pca_components is None or self.centroid is None:
            return {}
        x = np.asarray(vector, dtype=np.float64)
        comps = np.asarray(self.pca_components, dtype=np.float64)
        centered = x - np.asarray(self.centroid, dtype=np.float64)
        pcs = comps @ centered
        norm = np.linalg.norm(x) or 1.0
        centroid = np.asarray(self.centroid, dtype=np.float64)
        centroid_dist = 1.0 - float(np.dot(x / norm, centroid / (np.linalg.norm(centroid) or 1.0)))

        dists, _ = self._nn.kneighbors(x.reshape(1, -1))
        knn_dist = float(dists.mean())
        mean_knn_similarity = float((1.0 - dists).mean())

        lof_raw = -float(self._lof.score_samples(x.reshape(1, -1))[0])
        lof_score = (lof_raw - self.lof_offset) / self.lof_scale

        out = {
            "centroid_distance": centroid_dist,
            "knn_distance": knn_dist,
            "mean_knn_similarity": mean_knn_similarity,
            "lof_score": lof_score,
        }
        for i, val in enumerate(pcs.tolist()[:MAX_PCA_COMPONENTS], start=1):
            out[f"pc{i}"] = float(val)
        return out

    def to_artifact(self) -> dict[str, Any]:
        return {
            "pca_components": self.pca_components,
            "centroid": self.centroid,
            "knn_k": self.knn_k,
            "lof_offset": self.lof_offset,
            "lof_scale": self.lof_scale,
        }


# Deprecated alias — paper may still call these "novelty signals".
NoveltyModel = GeometryModel


@dataclass
class ZScoreModel:
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def fit(self, records: list[dict[str, Any]], calib_ids: set[str], keys: list[str]) -> None:
        buckets: dict[str, list[float]] = {k: [] for k in keys}
        for rec in records:
            if rec["query_id"] not in calib_ids:
                continue
            flat = flatten_blocks(rec)
            for key in keys:
                val = flat.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    buckets[key].append(float(val))
        for key in keys:
            vals = buckets[key]
            if not vals:
                self.mean[key] = 0.0
                self.std[key] = 1.0
            else:
                self.mean[key] = statistics.fmean(vals)
                self.std[key] = statistics.pstdev(vals) if len(vals) > 1 else 1.0

    def transform(self, record: dict[str, Any]) -> dict[str, Any]:
        scaled = _normalize_record_blocks(record)
        for block in JSONL_BLOCKS:
            if block not in scaled:
                continue
            block_out = dict(scaled[block])
            for key, val in list(block_out.items()):
                path = f"{block}.{key}"
                if path not in self.mean:
                    continue
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    continue
                std = self.std[path] or 1.0
                block_out[key] = (float(val) - self.mean[path]) / std
            scaled[block] = block_out
        return scaled

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}
