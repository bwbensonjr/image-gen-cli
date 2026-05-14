from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

from openai import OpenAI

MODEL = "gpt-5.5"
IMAGE_MODEL = "gpt-image-2"
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


def build_input(
    prompt: str,
    input_files: list[Path],
    file_id_map: dict[Path, str] | None = None,
) -> str | list[dict[str, Any]]:
    if not input_files:
        return prompt

    file_id_map = file_id_map or {}
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
        elif suffix == ".pdf":
            file_id = file_id_map.get(path)
            if file_id is None:
                raise ValueError(
                    f"PDF input must be uploaded via the Files API first: {path}"
                )
            parts.append({"type": "input_file", "file_id": file_id})
        elif _looks_like_image(suffix):
            raise ValueError(
                f"unsupported image type: {path.suffix}. Use png/jpg/webp/gif."
            )
        else:
            try:
                text = path.read_text()
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"cannot read {path} as text ({e.reason}). Supported binary "
                    f"inputs: png/jpg/webp/gif/pdf."
                ) from e
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
    verbose: bool = False,
    log_stream: TextIO | None = None,
    image_model: str = IMAGE_MODEL,
) -> list[Path]:
    api_client = client if client is not None else OpenAI()
    stream = log_stream or sys.stderr

    file_id_map = _upload_pdfs(api_client, input_files, verbose=verbose, stream=stream)
    try:
        payload = build_input(prompt, input_files, file_id_map=file_id_map)

        create_kwargs: dict[str, Any] = {
            "model": MODEL,
            "input": payload,
            "tools": [{"type": "image_generation", "model": image_model}],
        }
        if verbose:
            create_kwargs["reasoning"] = {"summary": "auto"}

        response = api_client.responses.create(**create_kwargs)
    finally:
        _delete_files(api_client, file_id_map.values(), verbose=verbose, stream=stream)

    if verbose:
        _log_response(response, log_stream or sys.stderr)

    images = [
        output.result
        for output in response.output
        if output.type == "image_generation_call" and output.result
    ]
    if not images:
        raise NoImageReturnedError(
            "model returned no image. Try rephrasing the prompt."
        )
    return write_images(images, output_base)


def _upload_pdfs(
    api_client: OpenAI,
    input_files: list[Path],
    verbose: bool,
    stream: TextIO,
) -> dict[Path, str]:
    file_id_map: dict[Path, str] = {}
    for path in input_files:
        if path.suffix.lower() != ".pdf":
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as fh:
            uploaded = api_client.files.create(file=fh, purpose="user_data")
        file_id_map[path] = uploaded.id
        if verbose:
            print(f"[upload] {path.name} -> {uploaded.id}", file=stream)
    return file_id_map


def _delete_files(
    api_client: OpenAI,
    file_ids: Iterable[str],
    verbose: bool,
    stream: TextIO,
) -> None:
    for file_id in file_ids:
        try:
            api_client.files.delete(file_id)
            if verbose:
                print(f"[cleanup] deleted {file_id}", file=stream)
        except Exception as e:
            if verbose:
                print(f"[cleanup] failed to delete {file_id}: {e}", file=stream)


def _log_response(response: Any, stream: TextIO) -> None:
    model = getattr(response, "model", None)
    status = getattr(response, "status", None)
    print(f"[response] model={model} status={status}", file=stream)

    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"[usage] {usage}", file=stream)

    for i, item in enumerate(response.output):
        _log_output_item(i, item, stream)


def _log_output_item(index: int, item: Any, stream: TextIO) -> None:
    item_type = getattr(item, "type", "?")
    item_id = getattr(item, "id", None)
    header = f"[output {index}] type={item_type}"
    if item_id:
        header += f" id={item_id}"

    if item_type == "reasoning":
        print(header, file=stream)
        for s in getattr(item, "summary", []) or []:
            print(f"  summary: {getattr(s, 'text', s)}", file=stream)
        for c in getattr(item, "content", []) or []:
            print(f"  reasoning: {getattr(c, 'text', c)}", file=stream)
    elif item_type == "image_generation_call":
        status = getattr(item, "status", None)
        result = getattr(item, "result", None)
        size = len(result) if result else 0
        print(f"{header} status={status} result_b64_len={size}", file=stream)
    elif item_type == "message":
        status = getattr(item, "status", None)
        print(f"{header} status={status}", file=stream)
        for c in getattr(item, "content", []) or []:
            text = getattr(c, "text", None)
            if text:
                print(f"  text: {text}", file=stream)
            else:
                print(f"  content: {c}", file=stream)
    else:
        print(f"{header} {item}", file=stream)
