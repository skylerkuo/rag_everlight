from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from rag_app.config import Settings
from rag_app.qa.engine import ANSWER_SYSTEM, RAGEngine, _context, _images
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
    """
    Read an existing output JSONL and return questions already completed.
    This allows --resume after interruption.
    """
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
    """
    Append one result immediately.
    Writing after every question prevents losing all results if the run stops.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Parse the model's JSON output.
    Also tolerates ```json ... ``` wrappers or extra surrounding text.
    """
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
    """
    Use the already-loaded Qwen3.5 model to extract:
      - keywords: general retrieval terms
      - proper_nouns: exact product/model/series names

    No extra model is loaded.
    """
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
    """
    Build searchable text from a SearchResult without depending on one
    specific SearchResult/Chunk implementation.
    """
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
    """
    Case-insensitive exact literal match with alphanumeric boundaries.

    This is intentionally strict:
    EL827 does not match EL8270.
    EL354N-G is matched as the complete literal string.
    """
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
    Exact Product Filter.

    If the LLM extracted one or more exact product/model/series names,
    keep only retrieved chunks that explicitly contain at least one of them.

    If no exact match exists in the candidate pool, fall back to the original
    retrieval results instead of returning an empty list.
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

def build_search_query(
    question: str,
    search_info: dict[str, list[str]],
) -> str:
    """
    Keep the original question and append extracted retrieval terms.

    proper_nouns are repeated in the query to strengthen exact lexical matching,
    while the same proper_nouns are also used later by Exact Product Filter.
    """
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

def entity_aware_ask(
    engine: RAGEngine,
    question: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Retrieval flow:
      1. Qwen extracts keywords + proper_nouns.
      2. Build expanded query.
      3. BGE-M3 retrieves a broader candidate pool.
      4. If proper_nouns exist, apply Exact Product Filter.
      5. Keep final Top-K results.
      6. Final answer still receives the ORIGINAL user question.
    """
    search_info = extract_search_info(engine, question)

    keywords = search_info.get("keywords", [])
    proper_nouns = search_info.get("proper_nouns", [])

    retrieval_query = build_search_query(
        question,
        search_info,
    )

    # Use a broader candidate pool so the exact-name filter has enough
    # candidates to work with.
    final_top_k = top_k if top_k is not None else 5
    candidate_top_k = max(final_top_k, 30)

    candidates = engine.index.search(
        retrieval_query,
        top_k=candidate_top_k,
    )

    filtered_candidates, filter_applied = apply_exact_product_filter(
        candidates,
        proper_nouns,
    )

    results = filtered_candidates[:final_top_k]

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
        "attached_pdf_images": [str(x) for x in images],
        "results": [x.to_dict() for x in results],
        "answer": answer,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch RAG evaluation runner: read questions from JSONL, "
            "run entity-aware RAG once per question, and save every question/answer."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path. Each line needs a `question` field.",
    )

    parser.add_argument(
        "--output",
        default="rag_model_outputs.jsonl",
        help="Output JSONL path. Default: rag_model_outputs.jsonl",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K retrieval results. Omit to use config default.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions for testing.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already successfully written to output.",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Also save full Top-K retrieval results for each question.",
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
    print("=" * 80)

    engine = RAGEngine(settings)

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total : {len(rows)}")
    print()

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
            )

            elapsed = time.perf_counter() - started

            record: dict[str, Any] = {
                "index": index,
                "category": source_row.get("category"),
                "question": question,

                # Ground-truth answer from the input JSONL, if present.
                "ground_truth": source_row.get("answer"),

                # Query analysis result.
                "keywords": result.get("keywords", []),
                "proper_nouns": result.get("proper_nouns", []),
                "exact_product_filter_applied": result.get(
                    "exact_product_filter_applied", False
                ),

                # Retrieval query actually sent to BGE-M3.
                "retrieval_query": result.get(
                    "retrieval_query", question
                ),

                # Actual answer produced by the RAG system.
                "model_answer": result.get("answer", ""),

                "attached_pdf_images": result.get(
                    "attached_pdf_images", []
                ),
                "elapsed_seconds": round(elapsed, 3),
                "status": "ok",
            }

            if args.save_results:
                record["retrieval_results"] = result.get(
                    "results", []
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
