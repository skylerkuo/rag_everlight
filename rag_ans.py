from __future__ import annotations

import argparse
import json
import math
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


class GeneratedTokenProbabilityCapture:
    """
    Capture generated-token probabilities from the SAME final VLM generation.

    Confidence:
        C_gen = exp(mean(log p(y_t | y_<t, x)))

    This is the geometric mean probability of the actually generated tokens.
    It is a raw generation-confidence score, not a calibrated probability that
    the factual answer is correct.

    Query-analyzer generation is NOT included. Only the final answer generation
    using ANSWER_SYSTEM is captured.
    """

    def __init__(self, answer_model: Any) -> None:
        self.answer_model = answer_model
        self.backend_model = self._find_backend_model(answer_model)

        if not hasattr(self.backend_model, "generate"):
            raise RuntimeError(
                "Could not find Hugging Face model.generate() on answer_model."
            )

        if not hasattr(self.backend_model, "compute_transition_scores"):
            raise RuntimeError(
                "The underlying Hugging Face model does not expose "
                "compute_transition_scores()."
            )

        self._original_answer_generate = answer_model.generate
        self._original_backend_generate = self.backend_model.generate

        self._capture_enabled = False
        self._installed = False
        self.last_stats: dict[str, Any] | None = None

    @staticmethod
    def _find_backend_model(answer_model: Any) -> Any:
        candidates = [
            getattr(answer_model, "model", None),
            getattr(answer_model, "_model", None),
            getattr(answer_model, "hf_model", None),
        ]

        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "generate"):
                return candidate

        raise RuntimeError(
            "Unable to locate the underlying Hugging Face Qwen model. "
            "Expected answer_model.model (or _model / hf_model)."
        )

    def install(self) -> None:
        if self._installed:
            return

        capture = self
        backend = self.backend_model

        def backend_generate_with_capture(*args: Any, **kwargs: Any) -> Any:
            if not capture._capture_enabled:
                return capture._original_backend_generate(*args, **kwargs)

            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True

            outputs = capture._original_backend_generate(*args, **kwargs)

            if not hasattr(outputs, "sequences") or not hasattr(outputs, "scores"):
                capture.last_stats = None
                return outputs

            sequences = outputs.sequences
            scores = outputs.scores

            if scores is None or len(scores) == 0:
                capture.last_stats = None
                return sequences

            beam_indices = getattr(outputs, "beam_indices", None)

            transition_scores = backend.compute_transition_scores(
                sequences,
                scores,
                beam_indices=beam_indices,
                normalize_logits=True,
            )

            log_probs = transition_scores[0].detach().float()
            log_probs = log_probs[log_probs.isfinite()]

            if log_probs.numel() == 0:
                capture.last_stats = None
                return sequences

            mean_log_prob = log_probs.mean().item()
            nll = -mean_log_prob
            generated_token_probability = math.exp(mean_log_prob)

            token_probs = log_probs.exp()

            capture.last_stats = {
                "generated_token_probability": float(
                    generated_token_probability
                ),
                "mean_log_probability": float(mean_log_prob),
                "negative_log_likelihood": float(nll),
                "perplexity": float(math.exp(nll)),
                "generated_token_count": int(log_probs.numel()),
                "mean_token_probability": float(
                    token_probs.mean().item()
                ),
                "min_token_probability": float(
                    token_probs.min().item()
                ),
            }

            # Preserve the behavior expected by the existing answer wrapper.
            return sequences

        def answer_generate_with_capture(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            is_final_answer = kwargs.get("system") == ANSWER_SYSTEM

            if is_final_answer:
                capture.last_stats = None
                capture._capture_enabled = True

            try:
                return capture._original_answer_generate(*args, **kwargs)
            finally:
                if is_final_answer:
                    capture._capture_enabled = False

        self.backend_model.generate = backend_generate_with_capture
        self.answer_model.generate = answer_generate_with_capture
        self._installed = True


def run_terminal(
    engine: RAGEngine,
    confidence_capture: GeneratedTokenProbabilityCapture,
    top_k: int | None,
) -> None:
    """
    Repeated stateless terminal queries.

    Each input is completely independent:
      - no conversation memory
      - no previous question/answer is reused
      - one retrieval pass only
      - no evidence-sufficiency loop
      - no query rewrite loop
    """

    print("\n" + "=" * 80)
    print("Everlight RAG Terminal")
    print("=" * 80)
    print("每一題都是獨立查詢，不保留前一題對話內容。")
    print("每題只做一次 Retrieval，不做三輪 Evidence 檢查 / Query Rewrite。")
    print("輸入 exit / quit / q 結束。")
    print("=" * 80)

    while True:
        try:
            question = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue

        if question.casefold() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        started = time.perf_counter()
        confidence_capture.last_stats = None

        try:
            result = entity_aware_ask(
                engine=engine,
                question=question,
                top_k=top_k,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print("\n" + "-" * 80)
            print("ERROR")
            print("-" * 80)
            print(f"{type(exc).__name__}: {exc}")
            print(f"Time: {elapsed:.2f} s")
            continue

        elapsed = time.perf_counter() - started
        stats = confidence_capture.last_stats

        print("\n" + "-" * 80)
        print("MODEL ANSWER")
        print("-" * 80)
        print(result.get("answer", ""))

        print("\n" + "-" * 80)
        print("CONFIDENCE")
        print("-" * 80)

        if stats is not None:
            confidence = stats["generated_token_probability"]
            print(
                "Generated-token probability: "
                f"{confidence * 100:.2f}%"
            )
        else:
            print("Generated-token probability: N/A")

        print("\n" + "-" * 80)
        print("RETRIEVAL STATUS")
        print("-" * 80)
        print(
            "Keywords: "
            + json.dumps(
                result.get("keywords", []),
                ensure_ascii=False,
            )
        )
        print(
            "Proper nouns: "
            + json.dumps(
                result.get("proper_nouns", []),
                ensure_ascii=False,
            )
        )
        print(
            "Exact Product Filter applied: "
            f"{result.get('exact_product_filter_applied', False)}"
        )
        print("Retrieval passes: 1")
        print(f"Time: {elapsed:.2f} s")
        print(
            "Note: Generated-token probability 是 raw generation confidence，"
            "不是校正後的答案正確率。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stateless Everlight RAG terminal: repeated independent questions, "
            "single retrieval pass, Exact Product Filter, and "
            "Generated-token probability only."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final Top-K retrieval results. Default: 5.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    setup_logging(args.verbose)

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
    print("Conversation memory: DISABLED.")
    print("Retrieval passes per question: 1.")
    print("Evidence-sufficiency iterative check: DISABLED.")
    print("Query rewrite loop: DISABLED.")
    print("Generated-token probability capture: ENABLED.")
    print("=" * 80)

    engine = RAGEngine(settings)

    confidence_capture = GeneratedTokenProbabilityCapture(
        engine.answer_model
    )
    confidence_capture.install()

    run_terminal(
        engine=engine,
        confidence_capture=confidence_capture,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
