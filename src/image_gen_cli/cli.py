from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from image_gen_cli import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-gen",
        description="Generate images from a prompt via the OpenAI Responses API.",
    )
    parser.add_argument("prompt", help="Description of the image to generate.")
    parser.add_argument(
        "-i",
        "--input-file",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Markdown/text or image file to attach as additional context. "
            "Repeatable. Image types: png/jpg/webp/gif; other extensions are "
            "treated as text."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output-image",
        metavar="BASE",
        help="Output filename base; images are written as BASE-1.png, BASE-2.png, ...",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full tracebacks on error.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"image-gen {__version__}",
    )
    return parser


def _print_error(message: str, verbose: bool) -> None:
    print(message, file=sys.stderr)
    if verbose:
        traceback.print_exc()
    else:
        print("(re-run with --verbose for the full traceback)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
        print(
            "error: OPENAI_API_KEY is not set. Export it, or set OPENAI_BASE_URL "
            "for a local proxy.",
            file=sys.stderr,
        )
        return 2

    input_paths = [Path(p) for p in args.input_file]
    for path in input_paths:
        if not path.exists():
            print(f"error: input file not found: {path}", file=sys.stderr)
            return 2

    print("Generating image...", file=sys.stderr)

    from image_gen_cli import generate

    try:
        written = generate.run(args.prompt, input_paths, args.output)
    except FileNotFoundError as e:
        _print_error(f"error: input file not found: {e}", args.verbose)
        return 2
    except ValueError as e:
        _print_error(f"error: {e}", args.verbose)
        return 2
    except generate.NoImageReturnedError as e:
        _print_error(f"error: {e}", args.verbose)
        return 1
    except Exception as e:
        _print_error(_map_openai_error(e), args.verbose)
        return 1

    for path in written:
        print(path)
    return 0


def _map_openai_error(e: Exception) -> str:
    try:
        from openai import (
            APIConnectionError,
            APIError,
            AuthenticationError,
            RateLimitError,
        )
    except ImportError:
        return f"error: {e}"

    if isinstance(e, AuthenticationError):
        return "error: OpenAI rejected the API key. Check OPENAI_API_KEY."
    if isinstance(e, RateLimitError):
        return "error: rate limited by OpenAI. Try again shortly."
    if isinstance(e, APIConnectionError):
        return "error: could not reach OpenAI API. Check network or OPENAI_BASE_URL."
    if isinstance(e, APIError):
        msg = getattr(e, "message", None) or str(e)
        return f"error: OpenAI API error: {msg}"
    return f"error: {e}"


if __name__ == "__main__":
    sys.exit(main())
