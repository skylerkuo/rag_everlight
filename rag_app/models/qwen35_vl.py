from __future__ import annotations

from pathlib import Path
from typing import Sequence
import tempfile

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class Qwen35VL:
    """Hugging Face Transformers inference for Qwen/Qwen3.5-4B."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        image_scale: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.image_scale = image_scale

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
        )

        self.model.eval()

    def _resize_image(
        self,
        image_path: str | Path,
        output_dir: Path,
    ) -> Path:
        image_path = Path(image_path)

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            new_width = max(1, int(img.width * self.image_scale))
            new_height = max(1, int(img.height * self.image_scale))

            img = img.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            output_path = output_dir / f"{image_path.stem}_resized.jpg"

            img.save(
                output_path,
                format="JPEG",
                quality=90,
            )

        return output_path

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] | None = None,
        image_labels: Sequence[str] | None = None,
        system: str | None = None,
        max_new_tokens: int = 1024,
    ) -> str:

        paths = list(image_paths or [])

        if image_labels is not None and len(image_labels) != len(paths):
            raise ValueError(
                "image_labels must have the same length as image_paths"
            )

        messages: list[dict] = []

        if system:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                        }
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            resized_paths = [
                self._resize_image(path, tmp_dir)
                for path in paths
            ]

            content: list[dict] = []

            for i, path in enumerate(resized_paths):

                if image_labels is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": image_labels[i],
                        }
                    )

                content.append(
                    {
                        "type": "image",
                        "path": str(path.resolve()),
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            generated_ids = output_ids[
                0,
                inputs["input_ids"].shape[-1] :
            ]

            return self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()