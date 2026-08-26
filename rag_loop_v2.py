from __future__ import annotations

import argparse
import json
import re
import time
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records. Each line must contain at least a `question` field."""
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no} must be a JSON object."
                )

            question = str(obj.get("question", "")).strip()
            if not question:
                raise ValueError(
                    f"Missing or empty `question` at line {line_no}."
                )

            rows.append(obj)

    return rows


def load_completed_questions(output_path: Path) -> set[str]:
    """Read an existing output JSONL and return completed questions."""
    completed: set[str] = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("status") == "ok":
                question = str(obj.get("question", "")).strip()
                if question:
                    completed.add(question)

    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one result immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON output, tolerating code fences or surrounding text."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
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
    """Extract keywords and exact product/model names with the loaded Qwen model."""
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


def _result_search_text(result: Any) -> str:
    """Build searchable text for exact product filtering."""
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


def _contains_exact_name(text: str, name: str) -> bool:
    """Case-insensitive literal match with alphanumeric boundaries."""
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
    """Keep candidates containing at least one exact extracted product/model name."""
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


def build_search_query(
    question: str,
    search_info: dict[str, list[str]],
) -> str:
    """Append extracted retrieval terms to the original question."""
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


def serialize_plain_results(
    results: list[Any],
    original_ranks: dict[int, int],
) -> list[dict[str, Any]]:
    """Serialize non-reranked results for baseline runs."""
    serialized: list[dict[str, Any]] = []

    for final_rank, result in enumerate(results, start=1):
        try:
            data = result.to_dict()
        except Exception:
            data = {"repr": repr(result)}

        if not isinstance(data, dict):
            data = {"result": data}

        data = dict(data)
        data["bge_rank_before_rerank"] = original_ranks.get(id(result))
        data["rerank_score"] = None
        data["rerank_rank"] = final_rank
        serialized.append(data)

    return serialized


def serialize_candidates(
    candidates: list[Any],
    original_ranks: dict[int, int],
) -> list[dict[str, Any]]:
    """Serialize the broad BGE-M3 candidate pool for debugging/evaluation."""
    output: list[dict[str, Any]] = []

    for result in candidates:
        try:
            data = result.to_dict()
        except Exception:
            data = {"repr": repr(result)}

        if not isinstance(data, dict):
            data = {"result": data}

        data = dict(data)
        data["bge_rank"] = original_ranks.get(id(result))
        output.append(data)

    return output


def entity_aware_ask(
    engine: RAGEngine,
    question: str,
    top_k: int | None = None,
    *,
    reranker: BGEReranker | None = None,
    candidate_top_k: int = 30,
) -> dict[str, Any]:
    """
    Retrieval flow:
    1. Extract keywords + exact product/model names.
    2. Build expanded retrieval query.
    3. BGE-M3 retrieves a broad candidate pool.
    4. Apply Exact Product Filter when possible.
    5. Rerank filtered candidates.
    6. Keep final Top-K and generate answer from the original question.
    """
    search_info = extract_search_info(engine, question)
    keywords = search_info.get("keywords", [])
    proper_nouns = search_info.get("proper_nouns", [])

    retrieval_query = build_search_query(
        question,
        search_info,
    )

    final_top_k = top_k if top_k is not None else 5
    candidate_top_k = max(final_top_k, candidate_top_k)

    retrieval_started = time.perf_counter()
    candidates = engine.index.search(
        retrieval_query,
        top_k=candidate_top_k,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started

    original_ranks = {
        id(result): rank
        for rank, result in enumerate(candidates, start=1)
    }

    filtered_candidates, filter_applied = apply_exact_product_filter(
        candidates,
        proper_nouns,
    )

    rerank_seconds = 0.0

    if reranker is not None:
        rerank_started = time.perf_counter()

        ranked_items = reranker.rerank(
            question,
            filtered_candidates,
            top_k=final_top_k,
            original_ranks=original_ranks,
        )

        rerank_seconds = time.perf_counter() - rerank_started
        results = [item["result"] for item in ranked_items]
        serialized_results = serialize_reranked_results(ranked_items)
    else:
        results = filtered_candidates[:final_top_k]
        serialized_results = serialize_plain_results(
            results,
            original_ranks,
        )

    candidate_results = serialize_candidates(
        candidates,
        original_ranks,
    )

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

    answer = engine.answer_model.generate(
        prompt,
        image_paths=images,
        system=ANSWER_SYSTEM,
        max_new_tokens=engine.settings.qwen_max_new_tokens_answer,
    )

    return {
        "question": question,
        "keywords": keywords,
        "proper_nouns": proper_nouns,
        "exact_product_filter_applied": filter_applied,
        "retrieval_query": retrieval_query,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "reranker_enabled": reranker is not None,
        "reranker_model": reranker.model_name if reranker is not None else None,
        "retrieval_seconds": round(retrieval_seconds, 3),
        "rerank_seconds": round(rerank_seconds, 3),
        "attached_pdf_images": [str(x) for x in images],
        "candidate_results": candidate_results,
        "results": serialized_results,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch entity-aware RAG evaluation with optional reranking."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path. Each line needs a `question` field.",
    )
    parser.add_argument(
        "--output",
        default="rag_model_outputs_v2.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Final Top-K after reranking. Default: 5.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=30,
        help="BGE-M3 candidate count before filtering/reranking.",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder reranker model.",
    )
    parser.add_argument(
        "--reranker-fp32",
        action="store_true",
        help="Disable FP16 for the reranker.",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranking for a baseline run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed questions.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save candidate and final retrieval results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    rows = load_jsonl(input_path)

    if args.limit is not None:
        rows = rows[: args.limit]

    completed = (
        load_completed_questions(output_path)
        if args.resume
        else set()
    )

    settings = Settings.fixed()
    path_problems = settings.validate_paths()

    if path_problems:
        raise SystemExit(
            "Configured dataset paths are invalid:\n- "
            + "\n- ".join(path_problems)
        )

    settings.ensure_dirs()

    print("=" * 80)
    print("Loading RAG engine...")
    print("BGE-M3 and Qwen3.5-VL are loaded only once.")
    print("Keyword + Proper-Noun extraction and Exact Product Filter are enabled.")

    if args.disable_reranker:
        print("Reranker: DISABLED")
    else:
        print(f"Reranker: {args.reranker_model}")

    print("=" * 80)

    engine = RAGEngine(settings)
    reranker: BGEReranker | None = None

    if not args.disable_reranker:
        print(f"Loading reranker: {args.reranker_model} ...")
        reranker = BGEReranker(
            model_name=args.reranker_model,
            use_fp16=not args.reranker_fp32,
        )
        print("Reranker loaded.")

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total : {len(rows)}")

    success = 0
    failed = 0
    skipped = 0

    for index, source_row in enumerate(rows, start=1):
        question = str(source_row["question"]).strip()

        if args.resume and question in completed:
            skipped += 1
            print(f"[{index}/{len(rows)}] SKIP: {question}")
            continue

        print("\n" + "=" * 80)
        print(f"[{index}/{len(rows)}]")
        print(f"Question: {question}")
        print("=" * 80)

        started = time.perf_counter()

        try:
            result = entity_aware_ask(
                engine,
                question,
                top_k=args.top_k,
                reranker=reranker,
                candidate_top_k=args.candidate_top_k,
            )
            elapsed = time.perf_counter() - started

            record: dict[str, Any] = {
                "index": index,
                "category": source_row.get("category"),
                "question": question,
                "ground_truth": source_row.get("answer"),
                "keywords": result.get("keywords", []),
                "proper_nouns": result.get("proper_nouns", []),
                "exact_product_filter_applied": result.get(
                    "exact_product_filter_applied",
                    False,
                ),
                "retrieval_query": result.get(
                    "retrieval_query",
                    question,
                ),
                "candidate_top_k": result.get("candidate_top_k"),
                "final_top_k": result.get("final_top_k"),
                "reranker_enabled": result.get("reranker_enabled", False),
                "reranker_model": result.get("reranker_model"),
                "retrieval_seconds": result.get("retrieval_seconds"),
                "rerank_seconds": result.get("rerank_seconds"),
                "model_answer": result.get("answer", ""),
                "attached_pdf_images": result.get(
                    "attached_pdf_images",
                    [],
                ),
                "elapsed_seconds": round(elapsed, 3),
                "status": "ok",
            }

            if args.save_results:
                record["retrieval_candidates_before_rerank"] = result.get(
                    "candidate_results",
                    [],
                )
                record["retrieval_results"] = result.get(
                    "results",
                    [],
                )

            append_jsonl(output_path, record)
            success += 1

            print("\nKEYWORDS:")
            print(
                json.dumps(
                    record["keywords"],
                    ensure_ascii=False,
                    indent=2,
                )
            )

            print("\nPROPER NOUNS / EXACT PRODUCTS:")
            print(
                json.dumps(
                    record["proper_nouns"],
                    ensure_ascii=False,
                    indent=2,
                )
            )

            print(
                "Exact Product Filter applied: "
                f"{record['exact_product_filter_applied']}"
            )

            print("\nRETRIEVAL QUERY:")
            print(record["retrieval_query"])

            print("\nRERANKER:")
            print(f"Enabled : {record['reranker_enabled']}")
            print(f"Model   : {record['reranker_model']}")
            print(f"Retrieve: {record['retrieval_seconds']} s")
            print(f"Rerank  : {record['rerank_seconds']} s")

            print("\nMODEL ANSWER:")
            print(record["model_answer"])
            print(f"\nTime: {record['elapsed_seconds']} s")
            print("Saved.")

        except Exception as exc:
            elapsed = time.perf_counter() - started

            record = {
                "index": index,
                "category": source_row.get("category"),
                "question": question,
                "ground_truth": source_row.get("answer"),
                "model_answer": None,
                "attached_pdf_images": [],
                "elapsed_seconds": round(elapsed, 3),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            append_jsonl(output_path, record)
            failed += 1

            print(f"\nERROR: {type(exc).__name__}: {exc}")
            print("Error record saved.")

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {output_path}")


if __name__ == "__main__":
    main()
