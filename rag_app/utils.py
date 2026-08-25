from __future__ import annotations

import gc
import hashlib
import logging
import re
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text


def strip_gemma_tokens(text: str) -> str:
    """Port of the cleanup behavior used by llm_eval's gemma_parser.py."""
    m = re.search(r"<\|turn>model\s*(.*?)\s*<turn\|>", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)", text, re.S)
    if m:
        return m.group(1).strip()
    return re.sub(r"<bos>|<eos>|<\|turn>[a-z]*|<turn\|>", "", text).strip()


def meaningful_text(text: str, min_chars: int = 60) -> bool:
    compact = re.sub(r"\s+", "", text)
    alnum = re.findall(r"[\w\u4e00-\u9fff]", compact, flags=re.UNICODE)
    return len(alnum) >= min_chars


def safe_yaml(value: str | None) -> str:
    if value is None:
        return '""'
    value = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{value}"'


def write_md_with_front_matter(path: Path, meta: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = safe_yaml(str(value))
        lines.append(f"{key}: {rendered}")
    lines.extend(["---", "", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def release_accelerator_memory(obj=None) -> None:
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
