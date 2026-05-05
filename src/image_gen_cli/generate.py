from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI

MODEL = "gpt-5.5"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class NoImageReturnedError(RuntimeError):
    """Raised when the model response contains no image_generation_call output."""


def build_input(prompt: str, input_files: list[Path]) -> str | list[dict[str, Any]]:
    if not input_files:
        return prompt

    parts: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            mime = MIME_BY_SUFFIX[suffix]
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append({
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{mime};base64,{b64}",
            })
        elif _looks_like_image(suffix):
            raise ValueError(
                f"unsupported image type: {path.suffix}. Use png/jpg/webp/gif."
            )
        else:
            text = path.read_text()
            parts.append({
                "type": "input_text",
                "text": f"<file: {path.name}>\n{text}",
            })

    return [{"role": "user", "type": "message", "content": parts}]


def _looks_like_image(suffix: str) -> bool:
    guess, _ = mimetypes.guess_type(f"x{suffix}")
    return bool(guess and guess.startswith("image/"))


def write_images(image_b64_list: list[str], output_base: str) -> list[Path]:
    paths: list[Path] = []
    for i, b64 in enumerate(image_b64_list, start=1):
        path = Path(f"{output_base}-{i}.png")
        path.write_bytes(base64.b64decode(b64))
        paths.append(path)
    return paths


def run(
    prompt: str,
    input_files: list[Path],
    output_base: str,
    client: OpenAI | None = None,
) -> list[Path]:
    api_client = client if client is not None else OpenAI()
    payload = build_input(prompt, input_files)
    response = api_client.responses.create(
        model=MODEL,
        input=payload,
        tools=[{"type": "image_generation"}],
    )
    images = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]
    if not images:
        raise NoImageReturnedError(
            "model returned no image. Try rephrasing the prompt."
        )
    return write_images(images, output_base)
