#!/usr/bin/env python3
"""Batch RAG evaluation for question-only JSONL.

Fair-test boundary
------------------
The input JSONL MAY contain evaluation-only fields such as:
    answer, ground_truth, source_title, page_number, source_chunk_id,
    source_url, source_excerpt, sources, origin, etc.

However, the RAG inference pipeline receives ONLY the `question` string.
Ground-truth answers and reference source locations are NEVER passed to:
    - query analysis
    - retrieval
    - exact-product filtering
    - reranking
    - final answer generation

Evaluation-only fields are copied to the output only AFTER that question has
finished inference, so they can be used later for offline correctness checking.

Pipeline
--------
1. Load Settings + RAGEngine directly.
2. Qwen extracts keywords + exact product/model names from the QUESTION only.
3. BGE-M3 retrieves candidate_top_k chunks.
4. Exact Product Filter is applied when possible.
5. Optional BGE reranker reranks candidates.
6. Keep final Top-K and log them.
7. Qwen generates the final answer.
8. Capture ONE confidence value:
       generated_token_probability
   = exp(mean(log p(generated token)))

This is raw generation confidence, NOT calibrated answer correctness probability.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_app.config import Settings
from rag_app.qa.engine import ANSWER_SYSTEM, RAGEngine, _context, _images
from rag_app.retrieval.reranker import BGEReranker, serialize_reranked_results
from rag_app.utils import setup_logging


ENTITY_SYSTEM = """You are a search query analyzer for a RAG system.

Extract the important search information from the user's question.

Return JSON only in exactly this structure:
{
  "keywords": [],
  "proper_nouns": []
}

Rules:
- keywords: important general search terms, technical properties, functions, conditions, or application concepts.
- proper_nouns: exact product names, model numbers, or product series names explicitly mentioned in the question.
- Preserve the original wording, spelling, capitalization, symbols, and punctuation.
- Only extract information that is present or clearly expressed in the question.
- Do not answer the question.
- Do not explain your reasoning.
- Do not invent, normalize, complete, or expand product names.
- If a category has no item, return an empty list.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Direct RAG evaluation: question-only JSONL -> retrieval/rerank -> "
            "final Top-K -> answer -> generated_token_probability."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Evaluation JSONL. Each row must contain a non-empty question; answer/source fields are allowed but never passed into RAG inference.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final Top-K evidence passed to the answer model. Default: 5.",
    )
    p.add_argument(
        "--candidate-top-k",
        type=int,
        default=30,
        help="BGE-M3 candidate count before exact filtering/reranking. Default: 30.",
    )
    p.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder reranker model.",
    )
    p.add_argument(
        "--reranker-fp32",
        action="store_true",
        help="Disable FP16 for reranker.",
    )
    p.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranking for a baseline run.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs already completed in output JSONL.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error.",
    )
    p.add_argument(
        "--max-content-chars",
        type=int,
        default=12000,
        help="Maximum stored characters per final Top-K chunk. 0 means unlimited.",
    )
    p.add_argument(
        "--save-candidates",
        action="store_true",
        help="Also store the BGE candidate pool before reranking.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return p.parse_args()


def validate_question_record(row: dict[str, Any], line_no: int) -> None:
    """
    Evaluation rows may contain answer/source/reference metadata.

    The only field required for RAG inference is `question`.
    Extra fields are intentionally NOT rejected because they are never passed
    into the RAG pipeline.
    """
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"Line {line_no}: missing non-empty question string."
        )


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue

            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_no}: expected a JSON object."
                )

            validate_question_record(row, line_no)
            rows.append(row)

    return rows


def read_completed_ids(output_path: Path) -> set[Any]:
    done: set[Any] = set()

    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            try:
                row = json.loads(raw)
            except Exception:
                continue

            if row.get("status") == "ok" and "id" in row:
                done.add(row["id"])

    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return {
        "keywords": [],
        "proper_nouns": [],
    }


def extract_search_info(
    engine: RAGEngine,
    question: str,
) -> dict[str, list[str]]:
    """Only the question is used to produce retrieval terms."""

    raw = engine.answer_model.generate(
        prompt=question,
        image_paths=None,
        system=ENTITY_SYSTEM,
        max_new_tokens=160,
    )

    parsed = _extract_json_object(raw)
    result: dict[str, list[str]] = {
        "keywords": [],
        "proper_nouns": [],
    }

    for key in ("keywords", "proper_nouns"):
        value = parsed.get(key, [])
        if not isinstance(value, list):
            value = []

        cleaned: list[str] = []
        seen: set[str] = set()

        for item in value:
            item = str(item).strip()
            if not item:
                continue

            normalized = item.casefold()
            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(item)

        result[key] = cleaned

    return result


def build_search_query(
    question: str,
    search_info: dict[str, list[str]],
) -> str:
    parts = [question]

    keywords = search_info.get("keywords", [])
    proper_nouns = search_info.get("proper_nouns", [])

    if keywords:
        parts.append(
            "Search keywords: "
            + " ; ".join(keywords)
        )

    if proper_nouns:
        parts.append(
            "Exact product/model names: "
            + " ; ".join(proper_nouns)
        )

    return "\n".join(parts)


def _result_search_text(result: Any) -> str:
    try:
        data = result.to_dict()
    except Exception:
        data = {}

    chunk = (
        data.get("chunk", data)
        if isinstance(data, dict)
        else {}
    )

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


def _contains_exact_name(text: str, name: str) -> bool:
    if not name:
        return False

    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def apply_exact_product_filter(
    results: list[Any],
    proper_nouns: list[str],
) -> tuple[list[Any], bool]:
    """
    Keep candidates explicitly containing at least one exact product/model name.

    If no exact match exists, fall back to original candidates.
    """

    if not proper_nouns:
        return results, False

    matched: list[Any] = []

    for result in results:
        text = _result_search_text(result)
        if any(
            _contains_exact_name(text, name)
            for name in proper_nouns
        ):
            matched.append(result)

    if matched:
        return matched, True

    return results, False


def to_plain(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(k): to_plain(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]

    if hasattr(value, "model_dump"):
        try:
            return to_plain(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return to_plain(value.dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return to_plain(vars(value))
        except Exception:
            pass

    return str(value)


def normalize_final_item(
    item: Any,
    rank: int,
    max_content_chars: int,
) -> dict[str, Any]:
    raw = to_plain(item)

    if not isinstance(raw, dict):
        return {
            "rank": rank,
            "content": str(raw),
        }

    # Reranker serializers often put the actual SearchResult under result/chunk.
    nested_result = (
        raw.get("result")
        if isinstance(raw.get("result"), dict)
        else {}
    )
    chunk = (
        raw.get("chunk")
        if isinstance(raw.get("chunk"), dict)
        else {}
    )
    if not chunk and isinstance(nested_result.get("chunk"), dict):
        chunk = nested_result["chunk"]

    metadata = {}
    for candidate in (
        raw.get("metadata"),
        nested_result.get("metadata"),
        chunk.get("metadata"),
    ):
        if isinstance(candidate, dict):
            metadata.update(candidate)

    def pick(*names: str) -> Any:
        for container in (raw, nested_result, chunk, metadata):
            if not isinstance(container, dict):
                continue
            for name in names:
                if name in container and container[name] is not None:
                    return container[name]
        return None

    content = pick(
        "content",
        "text",
        "page_content",
        "chunk_text",
        "embedding_text",
        "body",
    )

    if content is not None:
        content = str(content)
        if (
            max_content_chars > 0
            and len(content) > max_content_chars
        ):
            content = (
                content[:max_content_chars]
                + "…"
            )

    return {
        "rank": rank,
        "chunk_id": pick(
            "chunk_id",
            "id",
            "node_id",
            "passage_id",
        ),
        "document_id": pick(
            "document_id",
            "doc_id",
            "source_id",
        ),
        "title": pick(
            "title",
            "source_title",
            "document_title",
            "name",
        ),
        "page_number": pick(
            "page_number",
            "page",
            "page_no",
            "page_index",
        ),
        "source_url": pick(
            "source_url",
            "url",
            "canonical_url",
            "link",
        ),
        "score": pick(
            "score",
            "similarity",
            "similarity_score",
        ),
        "bge_rank_before_rerank": pick(
            "bge_rank_before_rerank",
            "bge_rank",
        ),
        "rerank_score": pick(
            "rerank_score",
            "reranker_score",
            "cross_encoder_score",
            "relevance_score",
        ),
        "rerank_rank": pick(
            "rerank_rank",
        ),
        "content": content,
    }


def serialize_candidates(
    candidates: list[Any],
    original_ranks: dict[int, int],
    max_content_chars: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for rank, result in enumerate(candidates, start=1):
        normalized = normalize_final_item(
            result,
            rank,
            max_content_chars,
        )
        normalized["bge_rank"] = original_ranks.get(
            id(result),
            rank,
        )
        output.append(normalized)

    return output


class GeneratedTokenProbabilityCapture:
    """
    Capture only one value from final Qwen answer generation:

        generated_token_probability
        = exp(mean(log p(y_t | y_<t, x)))

    Query-analyzer generation is excluded because capture is enabled only when
    system == ANSWER_SYSTEM.
    """

    def __init__(self, answer_model: Any) -> None:
        self.answer_model = answer_model
        self.backend_model = self._find_backend_model(
            answer_model
        )

        if not hasattr(
            self.backend_model,
            "compute_transition_scores",
        ):
            raise RuntimeError(
                "Underlying Hugging Face model does not expose "
                "compute_transition_scores()."
            )

        self._original_answer_generate = (
            answer_model.generate
        )
        self._original_backend_generate = (
            self.backend_model.generate
        )

        self._capture_enabled = False
        self._installed = False
        self.last_probability: float | None = None

    @staticmethod
    def _find_backend_model(
        answer_model: Any,
    ) -> Any:
        for candidate in (
            getattr(answer_model, "model", None),
            getattr(answer_model, "_model", None),
            getattr(answer_model, "hf_model", None),
        ):
            if (
                candidate is not None
                and hasattr(candidate, "generate")
            ):
                return candidate

        raise RuntimeError(
            "Unable to locate underlying Hugging Face model. "
            "Expected answer_model.model / _model / hf_model."
        )

    def install(self) -> None:
        if self._installed:
            return

        capture = self
        backend = self.backend_model

        def backend_generate_with_capture(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not capture._capture_enabled:
                return capture._original_backend_generate(
                    *args,
                    **kwargs,
                )

            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True

            outputs = capture._original_backend_generate(
                *args,
                **kwargs,
            )

            if (
                not hasattr(outputs, "sequences")
                or not hasattr(outputs, "scores")
            ):
                capture.last_probability = None
                return outputs

            sequences = outputs.sequences
            scores = outputs.scores

            if scores is None or len(scores) == 0:
                capture.last_probability = None
                return sequences

            beam_indices = getattr(
                outputs,
                "beam_indices",
                None,
            )

            transition_scores = (
                backend.compute_transition_scores(
                    sequences,
                    scores,
                    beam_indices=beam_indices,
                    normalize_logits=True,
                )
            )

            log_probs = (
                transition_scores[0]
                .detach()
                .float()
            )
            log_probs = log_probs[
                log_probs.isfinite()
            ]

            if log_probs.numel() == 0:
                capture.last_probability = None
                return sequences

            mean_log_prob = log_probs.mean().item()

            capture.last_probability = float(
                math.exp(mean_log_prob)
            )

            # Preserve existing answer wrapper behavior.
            return sequences

        def answer_generate_with_capture(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            is_final_answer = (
                kwargs.get("system")
                == ANSWER_SYSTEM
            )

            if is_final_answer:
                capture.last_probability = None
                capture._capture_enabled = True

            try:
                return capture._original_answer_generate(
                    *args,
                    **kwargs,
                )
            finally:
                if is_final_answer:
                    capture._capture_enabled = False

        self.backend_model.generate = (
            backend_generate_with_capture
        )
        self.answer_model.generate = (
            answer_generate_with_capture
        )
        self._installed = True

    def reset(self) -> None:
        self.last_probability = None


def entity_aware_ask(
    engine: RAGEngine,
    confidence_capture: GeneratedTokenProbabilityCapture,
    question: str,
    *,
    top_k: int,
    candidate_top_k: int,
    reranker: BGEReranker | None,
    max_content_chars: int,
    save_candidates: bool,
) -> dict[str, Any]:
    """
    Direct RAG flow. Only question-derived information is used.
    """

    # Query analysis is NOT included in final-answer confidence.
    search_info = extract_search_info(
        engine,
        question,
    )

    keywords = search_info.get(
        "keywords",
        [],
    )
    proper_nouns = search_info.get(
        "proper_nouns",
        [],
    )

    retrieval_query = build_search_query(
        question,
        search_info,
    )

    candidate_top_k = max(
        top_k,
        candidate_top_k,
    )

    retrieval_started = time.perf_counter()

    candidates = engine.index.search(
        retrieval_query,
        top_k=candidate_top_k,
    )

    retrieval_seconds = (
        time.perf_counter()
        - retrieval_started
    )

    original_ranks = {
        id(result): rank
        for rank, result
        in enumerate(candidates, start=1)
    }

    filtered_candidates, filter_applied = (
        apply_exact_product_filter(
            candidates,
            proper_nouns,
        )
    )

    rerank_seconds = 0.0

    if reranker is not None:
        rerank_started = time.perf_counter()

        ranked_items = reranker.rerank(
            question,
            filtered_candidates,
            top_k=top_k,
            original_ranks=original_ranks,
        )

        rerank_seconds = (
            time.perf_counter()
            - rerank_started
        )

        results = [
            item["result"]
            for item in ranked_items
        ]

        raw_final_results = (
            serialize_reranked_results(
                ranked_items
            )
        )
    else:
        results = filtered_candidates[:top_k]

        raw_final_results = []
        for final_rank, result in enumerate(
            results,
            start=1,
        ):
            try:
                data = result.to_dict()
            except Exception:
                data = {
                    "repr": repr(result),
                }

            if not isinstance(data, dict):
                data = {
                    "result": data,
                }

            data = dict(data)
            data["bge_rank_before_rerank"] = (
                original_ranks.get(
                    id(result)
                )
            )
            data["rerank_score"] = None
            data["rerank_rank"] = final_rank
            raw_final_results.append(data)

    final_top_k = [
        normalize_final_item(
            item,
            rank,
            max_content_chars,
        )
        for rank, item in enumerate(
            raw_final_results,
            start=1,
        )
    ]

    images = _images(
        results,
        engine.settings.max_answer_images,
    )

    prompt = f"""USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{_context(results)}

Instructions:
- Answer the user question using the evidence above.
- If PDF page images are attached, use them to verify tables, figures, values, labels,
  or layout details that may not be fully represented in the extracted Markdown.
- Cite evidence with [S1], [S2], etc.
"""

    confidence_capture.reset()

    generation_started = time.perf_counter()

    answer = engine.answer_model.generate(
        prompt,
        image_paths=images,
        system=ANSWER_SYSTEM,
        max_new_tokens=(
            engine.settings
            .qwen_max_new_tokens_answer
        ),
    )

    generation_seconds = (
        time.perf_counter()
        - generation_started
    )

    output: dict[str, Any] = {
        "keywords": keywords,
        "proper_nouns": proper_nouns,
        "exact_product_filter_applied": (
            filter_applied
        ),
        "retrieval_query": retrieval_query,
        "candidate_top_k": candidate_top_k,
        "final_top_k_count": len(final_top_k),
        "reranker_enabled": (
            reranker is not None
        ),
        "reranker_model": (
            reranker.model_name
            if reranker is not None
            else None
        ),
        "retrieval_seconds": round(
            retrieval_seconds,
            3,
        ),
        "rerank_seconds": round(
            rerank_seconds,
            3,
        ),
        "generation_seconds": round(
            generation_seconds,
            3,
        ),
        "model_answer": answer,
        "final_top_k": final_top_k,
        "generated_token_probability": (
            confidence_capture
            .last_probability
        ),
        "attached_pdf_images": [
            str(x)
            for x in images
        ],
    }

    if save_candidates:
        output["retrieval_candidates_before_rerank"] = (
            serialize_candidates(
                candidates,
                original_ranks,
                max_content_chars,
            )
        )

    return output



def extract_evaluation_reference(row: dict[str, Any]) -> dict[str, Any]:
    """
    Return all non-runtime evaluation metadata from the input row.

    IMPORTANT:
    This function is used only AFTER RAG inference for logging/offline scoring.
    Its return value is never passed to retrieval, reranking, or generation.
    """
    excluded = {"id", "category", "question"}
    return {
        str(key): value
        for key, value in row.items()
        if key not in excluded
    }



def main() -> int:
    args = parse_args()

    if args.top_k < 1:
        raise SystemExit(
            "--top-k must be >= 1"
        )

    if args.candidate_top_k < 1:
        raise SystemExit(
            "--candidate-top-k must be >= 1"
        )

    input_path = (
        Path(args.input)
        .expanduser()
        .resolve()
    )
    output_path = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    if not input_path.exists():
        raise SystemExit(
            f"Input JSONL not found: {input_path}"
        )

    questions = load_questions(
        input_path
    )

    if args.limit is not None:
        questions = questions[
            :args.limit
        ]

    completed = (
        read_completed_ids(output_path)
        if args.resume
        else set()
    )

    setup_logging(args.verbose)

    settings = Settings.fixed()

    path_problems = (
        settings.validate_paths()
    )

    if path_problems:
        raise SystemExit(
            "Configured dataset paths are invalid:\n- "
            + "\n- ".join(path_problems)
        )

    settings.ensure_dirs()

    print("=" * 80)
    print("Loading RAG engine...")
    print("Input mode: FULL EVALUATION JSONL ALLOWED")
    print("RAG inference input: QUESTION ONLY")
    print("Ground truth/reference metadata: ISOLATED FROM RAG")
    print(
        f"Candidate Top-K: "
        f"{args.candidate_top_k}"
    )
    print(
        f"Final Top-K: "
        f"{args.top_k}"
    )
    print("=" * 80)

    engine = RAGEngine(settings)

    confidence_capture = (
        GeneratedTokenProbabilityCapture(
            engine.answer_model
        )
    )
    confidence_capture.install()

    reranker: BGEReranker | None = None

    if not args.disable_reranker:
        print(
            "Loading reranker: "
            f"{args.reranker_model}"
        )

        reranker = BGEReranker(
            model_name=args.reranker_model,
            use_fp16=(
                not args.reranker_fp32
            ),
        )

        print("Reranker loaded.")
    else:
        print("Reranker: DISABLED")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total : {len(questions)}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.resume:
        output_path.write_text(
            "",
            encoding="utf-8",
        )

    success = 0
    failed = 0
    skipped = 0

    for seq, row in enumerate(
        questions,
        start=1,
    ):
        qid = row.get("id", seq)

        if (
            args.resume
            and qid in completed
        ):
            skipped += 1
            print(
                f"[{seq}/{len(questions)}] "
                f"SKIP id={qid}"
            )
            continue

        # FAIR-TEST BOUNDARY:
        # From this point through entity_aware_ask(), ONLY `question` is used by RAG.
        # answer/source/reference fields remain in `row` but are not passed anywhere.
        question = str(row["question"]).strip()

        print("\n" + "=" * 80)
        print(
            f"[{seq}/{len(questions)}] "
            f"id={qid}"
        )
        print(
            f"Question: {question}"
        )
        print("=" * 80)

        started = time.perf_counter()

        record: dict[str, Any] = {
            "id": qid,
            "category": row.get(
                "category"
            ),
            "question": question,
            "status": "ok",
            "logged_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        try:
            result = entity_aware_ask(
                engine=engine,
                confidence_capture=(
                    confidence_capture
                ),
                question=question,
                top_k=args.top_k,
                candidate_top_k=(
                    args.candidate_top_k
                ),
                reranker=reranker,
                max_content_chars=(
                    args.max_content_chars
                ),
                save_candidates=(
                    args.save_candidates
                ),
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            record.update(result)
            record["elapsed_seconds"] = round(
                elapsed,
                3,
            )

            # Evaluation-only fields are attached ONLY AFTER inference is complete.
            # They were not visible to Query Analyzer / Retrieval / Reranker / Qwen.
            evaluation_reference = extract_evaluation_reference(row)
            if evaluation_reference:
                record["evaluation_reference"] = evaluation_reference

            success += 1

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - started
            )

            record.update(
                {
                    "status": "error",
                    "model_answer": None,
                    "final_top_k": [],
                    "generated_token_probability": None,
                    "elapsed_seconds": round(
                        elapsed,
                        3,
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                    "traceback": (
                        traceback.format_exc(
                            limit=8
                        )
                    ),
                }
            )

            evaluation_reference = extract_evaluation_reference(row)
            if evaluation_reference:
                record["evaluation_reference"] = evaluation_reference

            failed += 1

            if args.fail_fast:
                append_jsonl(
                    output_path,
                    record,
                )
                raise

        append_jsonl(
            output_path,
            record,
        )

        probability = record.get(
            "generated_token_probability"
        )
        probability_text = (
            "N/A"
            if probability is None
            else f"{float(probability):.6f}"
        )

        print(
            f"status={record['status']} "
            f"final_top_k="
            f"{len(record.get('final_top_k', []))} "
            f"generated_token_probability="
            f"{probability_text} "
            f"time="
            f"{record.get('elapsed_seconds')}s"
        )

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())