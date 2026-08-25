from __future__ import annotations

from pathlib import Path

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.utils import strip_gemma_tokens


class Gemma4GGUF:
    """Gemma 4 E4B local inference using the same style as llm_eval-main.

    llm_eval-main uses ``ChatLlamaCpp(...).invoke([SystemMessage, HumanMessage])``.
    This wrapper keeps that exact integration pattern while making the GGUF path and
    context settings configurable for document preprocessing.
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        temperature: float = 0.0,
        max_tokens: int = 1800,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Gemma 4 E4B GGUF not found: {model_path}\n"
                "Edit GEMMA_MODEL_PATH at the top of rag_app/config.py to point "
                "to gemma-4-E4B-it-Q4_K_M.gguf."
            )
        self.chat_model = ChatLlamaCpp(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            f16_kv=True,
            verbose=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, system: str, user: str) -> str:
        response = self.chat_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return strip_gemma_tokens((response.content or "").strip())
