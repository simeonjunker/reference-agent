# reference-agent

A small LangGraph system that describes a **target** image so it can be distinguished from two **distractor** images, using a vision-language model served by an OpenAI-compatible API.

Graph: `validate -> describe`.

## Setup

    uv sync

Configuration is read from environment variables or a `.env` file (copy `.env.example` to `.env`; it is git-ignored). Real environment variables take precedence. All are optional: `VLM_BASE_URL` (default `http://129.70.40.28:8345/v1`), `VLM_MODEL` (default `unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`), `VLM_API_KEY` (default `none`).

## Contract

- Input: each image is a PNG/JPEG file path (`str`/`Path`) or a `PIL.Image.Image`, under the keys `target`, `distractor1`, `distractor2`.
- Output: final state; the description is `result["description"]` (str).
- Missing keys, missing files or other formats raise `ValueError`/`FileNotFoundError`; model errors propagate.

## Usage

Shortest form: `describe_target(target, distractor1, distractor2, model=None)` returns the description as `str`.

A demo notebook is in `notebooks/demo.ipynb` (placeholder images come from `notebooks/make_images.py`).

    from reference_agent.graph import build_graph
    result = build_graph().invoke({"target": "t.png", "distractor1": "d1.png", "distractor2": "d2.png"})
    print(result["description"])

CLI: `uv run reference-agent t.png d1.png d2.png`

## Tests

    uv run pytest

Tests use a fake model; no network needed.
