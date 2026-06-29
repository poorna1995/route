"""φ_semantic encoder and R_c-only engineering (φ_novelty, z-score)."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_routing.query_derived.config import JSONL_BLOCKS, flatten_blocks, novelty_section


# --- φ_semantic ---


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


# --- φ_novelty ---


@dataclass
class NoveltyModel:
    pca_components: list[list[float]] | None = None
    centroid: list[float] | None = None
    knn_k: int = 10
    lof_offset: float = 0.0
    lof_scale: float = 1.0

    def fit(self, matrix: list[list[float]], config: dict[str, Any]) -> None:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

        ncfg = novelty_section(config)
        n_comp = int(ncfg.get("pca_components", 3))
        k = int(ncfg.get("knn_k", 10))

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
        retrieval_density = float((1.0 - dists).mean())

        lof_raw = -float(self._lof.score_samples(x.reshape(1, -1))[0])
        lof_score = (lof_raw - self.lof_offset) / self.lof_scale

        out = {
            "centroid_distance": centroid_dist,
            "knn_distance": knn_dist,
            "retrieval_density": retrieval_density,
            "lof_score": lof_score,
        }
        for i, val in enumerate(pcs.tolist(), start=1):
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


GeometryModel = NoveltyModel


# --- z-score (implementation detail) ---


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
        scaled = dict(record)
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
