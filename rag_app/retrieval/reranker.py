from __future__ import annotations

from typing import Any


def _result_search_text(result: Any) -> str:
    """Build generic searchable text from a SearchResult-like object."""
    try:
        data = result.to_dict()
    except Exception:
        data = {}

    chunk = data.get("chunk", data) if isinstance(data, dict) else {}
    strings: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(chunk)
    return "\n".join(strings)


def _result_rerank_text(result: Any) -> str:
    """Build a concise passage for cross-encoder reranking."""
    try:
        data = result.to_dict()
    except Exception:
        data = {}

    chunk = data.get("chunk", data) if isinstance(data, dict) else {}
    if not isinstance(chunk, dict):
        return _result_search_text(result)

    parts: list[str] = []

    def add(label: str, value: Any) -> None:
        if value is None:
            return

        if isinstance(value, str):
            value = value.strip()
            if value:
                parts.append(f"{label}: {value}" if label else value)
            return

        if isinstance(value, list):
            items = [str(x).strip() for x in value if str(x).strip()]
            if items:
                joined = " > ".join(items)
                parts.append(f"{label}: {joined}" if label else joined)

    add("Title", chunk.get("title") or chunk.get("document_title"))
    add("Heading", chunk.get("heading_path") or chunk.get("heading"))
    add("Section", chunk.get("section"))

    content = (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("markdown")
        or chunk.get("body")
        or chunk.get("page_text")
    )
    add("Content", content)

    if parts:
        return "\n".join(parts)

    return _result_search_text(result)


class BGEReranker:
    """Second-stage cross-encoder reranker using BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is required for reranking. "
                "Install it with: pip install -U FlagEmbedding"
            ) from exc

        self.model_name = model_name
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16,
        )

    def rerank(
        self,
        query: str,
        candidates: list[Any],
        top_k: int,
        original_ranks: dict[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank SearchResult-like objects.

        Returns dictionaries containing:
        - result
        - bge_rank
        - filtered_candidate_rank
        - rerank_score
        - rerank_rank
        """
        if not candidates:
            return []

        pairs = [
            [query, _result_rerank_text(result)]
            for result in candidates
        ]

        scores = self.model.compute_score(
            pairs,
            normalize=True,
        )

        if isinstance(scores, (int, float)):
            scores = [scores]

        ranked: list[dict[str, Any]] = []

        for filtered_rank, (result, score) in enumerate(
            zip(candidates, scores),
            start=1,
        ):
            bge_rank = (
                original_ranks.get(id(result), filtered_rank)
                if original_ranks
                else filtered_rank
            )

            ranked.append(
                {
                    "result": result,
                    "bge_rank": int(bge_rank),
                    "filtered_candidate_rank": filtered_rank,
                    "rerank_score": float(score),
                }
            )

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )
        ranked = ranked[:top_k]

        for rerank_rank, item in enumerate(ranked, start=1):
            item["rerank_rank"] = rerank_rank

        return ranked


def serialize_reranked_results(
    ranked_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize final results while preserving retrieval and rerank ranks."""
    output: list[dict[str, Any]] = []

    for item in ranked_items:
        result = item["result"]

        try:
            data = result.to_dict()
        except Exception:
            data = {"repr": repr(result)}

        if not isinstance(data, dict):
            data = {"result": data}

        data = dict(data)
        data["bge_rank_before_rerank"] = item.get("bge_rank")
        data["filtered_candidate_rank"] = item.get("filtered_candidate_rank")
        data["rerank_score"] = item.get("rerank_score")
        data["rerank_rank"] = item.get("rerank_rank")
        output.append(data)

    return output
