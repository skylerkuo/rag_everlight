from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_app.config import Settings
from rag_app.metadata import read_jsonl

LOGGER = logging.getLogger(__name__)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-12)


def _sparse_dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    # FlagEmbedding may serialize token ids as either int or str.
    b2 = {str(k): float(v) for k, v in b.items()}
    return float(sum(float(v) * b2.get(str(k), 0.0) for k, v in a.items()))


def _rrf(rank_lists: list[list[int]], weights: list[float], k0: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking, weight in zip(rank_lists, weights):
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + weight / (k0 + rank)
    return scores


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk: dict
    dense_score: float | None = None
    sparse_score: float | None = None
    bge_pair_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "bge_pair_score": self.bge_pair_score,
            **self.chunk,
        }


class BGEM3Index:
    def __init__(self, settings: Settings, load_model: bool = True) -> None:
        self.settings = settings
        self.chunks: list[dict] = []
        self.dense: np.ndarray | None = None
        self.sparse: list[dict] = []
        self.model = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        if self.model is None:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                self.settings.bge_model_id,
                use_fp16=self.settings.bge_use_fp16,
            )

    @property
    def dense_path(self) -> Path:
        return self.settings.index_dense_path

    @property
    def sparse_path(self) -> Path:
        return self.settings.index_sparse_path

    @property
    def chunk_snapshot_path(self) -> Path:
        return self.settings.index_chunks_path

    @property
    def meta_path(self) -> Path:
        return self.settings.index_meta_path

    def build(self) -> dict:
        self.settings.ensure_dirs()
        self._load_model()
        chunks = read_jsonl(self.settings.chunks_path)
        if not chunks:
            raise RuntimeError(f"No chunks found at {self.settings.chunks_path}; run chunk first.")
        texts = [c["embedding_text"] for c in chunks]
        LOGGER.info("Encoding %s chunks with %s", len(texts), self.settings.bge_model_id)
        out = self.model.encode(
            texts,
            batch_size=self.settings.bge_batch_size,
            max_length=self.settings.bge_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = _normalize_rows(np.asarray(out["dense_vecs"], dtype=np.float32))
        sparse = out["lexical_weights"]
        np.save(self.dense_path, dense)
        with self.sparse_path.open("w", encoding="utf-8") as f:
            for row in sparse:
                f.write(json.dumps({str(k): float(v) for k, v in row.items()}) + "\n")
        with self.chunk_snapshot_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        self.meta_path.write_text(
            json.dumps(
                {
                    "model": self.settings.bge_model_id,
                    "chunks": len(chunks),
                    "dense_dim": int(dense.shape[1]),
                    "max_length": self.settings.bge_max_length,
                    "retrieval": "dense+sparse RRF; optional BGE-M3 pair scoring on candidates",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.chunks, self.dense, self.sparse = chunks, dense, sparse
        return {"chunks": len(chunks), "dense_dim": int(dense.shape[1]), "index_dir": str(self.settings.index_dir)}

    def load(self) -> None:
        if not self.chunk_snapshot_path.exists() or not self.dense_path.exists() or not self.sparse_path.exists():
            raise RuntimeError("Index files are missing; run `python rag.py index ...` first.")
        self.chunks = read_jsonl(self.chunk_snapshot_path)
        self.dense = np.load(self.dense_path)
        self.sparse = read_jsonl(self.sparse_path)
        self._load_model()

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        if self.dense is None or not self.chunks:
            self.load()
        assert self.dense is not None
        assert self.model is not None
        top_k = top_k or self.settings.top_k
        candidate_k = min(max(self.settings.candidate_k, top_k), len(self.chunks))

        q = self.model.encode(
            [query],
            batch_size=1,
            max_length=self.settings.bge_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        q_dense = _normalize_vec(np.asarray(q["dense_vecs"][0], dtype=np.float32))
        q_sparse = q["lexical_weights"][0]

        dense_scores = self.dense @ q_dense
        sparse_scores = np.asarray([_sparse_dot(q_sparse, x) for x in self.sparse], dtype=np.float32)

        dense_rank = np.argsort(-dense_scores)[:candidate_k].tolist()
        sparse_rank = np.argsort(-sparse_scores)[:candidate_k].tolist()
        fused = _rrf([dense_rank, sparse_rank], weights=[0.4, 0.6])
        candidates = sorted(fused, key=fused.get, reverse=True)[:candidate_k]

        pair_scores: dict[int, float] = {}
        if self.settings.use_bge_pair_rerank and candidates:
            try:
                pairs = [[query, self.chunks[i]["embedding_text"]] for i in candidates]
                score_out = self.model.compute_score(
                    pairs,
                    max_passage_length=self.settings.bge_max_length,
                    weights_for_different_modes=[0.4, 0.2, 0.4],
                )
                vals = score_out.get("colbert+sparse+dense") or score_out.get("dense")
                if vals is not None:
                    pair_scores = {idx: float(v) for idx, v in zip(candidates, vals)}
                    candidates = sorted(candidates, key=lambda i: pair_scores[i], reverse=True)
            except Exception as exc:
                LOGGER.warning("BGE pair scoring unavailable; using hybrid RRF only: %s", exc)

        final = candidates[:top_k]
        results: list[SearchResult] = []
        for rank, idx in enumerate(final, 1):
            score = pair_scores.get(idx, fused.get(idx, 0.0))
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    chunk=self.chunks[idx],
                    dense_score=float(dense_scores[idx]),
                    sparse_score=float(sparse_scores[idx]),
                    bge_pair_score=pair_scores.get(idx),
                )
            )
        return results
