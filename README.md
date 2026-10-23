# reference-agent

A small LangGraph system that describes a **target** image so it can be distinguished from two **distractor** images, using a vision-language model served by an OpenAI-compatible API.

Graph: `validate -> describe -> internal_listener_eval`, then either the end or `refine -> internal_listener_eval` (loop).

- `internal_listener_eval`: the same model sees the three images in random order, unlabeled, and picks the one the description refers to.
- `refine`: runs only if the listener did not pick the target; rewrites the description, then the listener is asked again.
- Simple variant: `build_graph(use_listener=False)` / `describe_target(..., use_listener=False)` builds `validate -> describe` only; it ends right after the initial description (one model call, no `attempts`/`listener_choice` in the result).
- After 10 evaluations without success (`MAX_EVAL_ROUNDS` in `graph.py`) a `RuntimeError` is raised.

## Setup

    uv sync

Configuration is read from environment variables or a `.env` file (copy `.env.example` to `.env`; it is git-ignored). Real environment variables take precedence. All are optional: `VLM_BASE_URL` (default `http://129.70.40.28:8345/v1`), `VLM_MODEL` (default `unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`), `VLM_API_KEY` (default `none`), `VLM_TIMEOUT` in seconds per request (default `300`), `VLM_MAX_RETRIES` (default `0`).

## Contract

- Input: each image is a PNG/JPEG file path (`str`/`Path`) or a `PIL.Image.Image`, under the keys `target`, `distractor1`, `distractor2`.
- Output: final state; the description is `result["description"]` (str).
- Missing keys, missing files or other formats raise `ValueError`/`FileNotFoundError`; model errors propagate. A listener answer that is not 1, 2 or 3 raises a `ValueError`. The final state also contains `attempts` and `listener_choice`.

## Usage

Shortest form: `describe_target(target, distractor1, distractor2, model=None)` returns the description as `str`.

A demo notebook is in `notebooks/demo.ipynb` (placeholder images come from `notebooks/make_images.py`).

    from reference_agent.graph import build_graph
    result = build_graph().invoke({"target": "t.png", "distractor1": "d1.png", "distractor2": "d2.png"})
    print(result["description"])

CLI: `uv run reference-agent [--no-listener] t.png d1.png d2.png`

## Tests

    uv run pytest

Tests use a fake model; no network needed.
