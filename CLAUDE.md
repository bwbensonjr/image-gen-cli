# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Command-line tool for generating images via the OpenAI API (initial target model: `gpt-image-2`). The CLI accepts a prompt plus optional supporting inputs (markdown/text files, example images) and writes one or more PNGs using a configurable output filename spec (default base: `output-image`, producing files like `output-image-1.png`).

The current `src/imagegen.py` is an early sketch — it hardcodes the prompt from `sys.argv[1]` and writes to a fixed path. Treat it as scaffolding to be replaced, not a working baseline. (Note: the model string `gpt-5.5` in both the sketch and the README example is intentional — see the invocation pattern below.)

## Invoking the Image Model

Per the README, `gpt-image-2` is **not** called via a dedicated images endpoint. It is invoked as a **tool** through the Responses API on a chat model. Follow this pattern:

```python
response = client.responses.create(
    model="gpt-5.5",
    input=prompt,
    tools=[{"type": "image_generation"}],
)
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]
# image_data[i] is base64-encoded PNG bytes — decode with base64.b64decode before writing.
```

Key implications for the CLI:
- The driver model goes in `model=`; the image generation capability is enabled by the `image_generation` tool entry. Don't switch to `client.images.generate(...)`.
- Walk `response.output` and filter on `output.type == "image_generation_call"` to collect results — there can be more than one, which maps directly to the numbered output files.
- Each `output.result` is base64; always `base64.b64decode` before writing the `.png`.
- Optional inputs from the README (markdown/text files, example images) should be threaded into the `input=` payload following the Responses API multimodal input shape, not as separate API calls.

## Commands

- `uv sync` — install/refresh dependencies from `uv.lock`.
- `uv run python src/imagegen.py "<prompt>"` — run the CLI in the project venv.
- `uv add <pkg>` / `uv remove <pkg>` — manage dependencies (do **not** edit `pyproject.toml` deps by hand or use `pip`).

No test, lint, or build configuration exists yet.

## Design Constraints (from README)

- Use the **OpenAI Python SDK** and rely on its environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) — do not add custom config for these.
- Use the standard library **`argparse`** for argument parsing (not click/typer).
- Follow the [clig.dev](https://clig.dev) command-line guidelines, which are cached locally at `docs/cli_guidelines.md`. Consult that file when designing flags, output behavior, error messages, and exit codes.

## Output Filename Convention

When the user supplies `--output foo` (or whatever the flag ends up named), generated images must be written as `foo-1.png`, `foo-2.png`, … — always numbered, even for a single image, so the scheme is consistent across calls.
