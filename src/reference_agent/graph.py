import base64
import io
import os
import random
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from PIL import Image
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Fixed order in which the images are sent to the model: target first, then both distractors.
IMAGE_KEYS = ("target", "distractor1", "distractor2")
LABELS = {"target": "Target", "distractor1": "Distractor 1", "distractor2": "Distractor 2"}
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

INSTRUCTION = """\
You see three images: the target, distractor 1 and distractor 2. \
Describe the target in one or two short sentences so that someone could pick it out \
from the two distractors. Mention only visible features that distinguish the target \
from both distractors, and make no assumptions beyond what is visible."""

REFINE_INSTRUCTION = """\
A listener who saw the three images without labels read the description below and wrongly \
picked {chosen} as the described image, so it is not discriminative enough. \
Rewrite the description of the target in one or two short sentences, using visible features \
that clearly set the target apart from both distractors. Reply with the new description only."""

LISTENER_INSTRUCTION = """\
Below are three images and a description of exactly one of them. \
Which image does the description refer to? Answer with only the number 1, 2 or 3."""

# Upper bound on listener evaluations; reaching it without success raises an error.
MAX_EVAL_ROUNDS = 10

ImageInput = str | Path | Image.Image


# Graph state: the three image inputs, the current description candidate, the role the listener
# picked for it, and the number of listener evaluations so far.
class State(TypedDict, total=False):
    target: ImageInput
    distractor1: ImageInput
    distractor2: ImageInput
    description: str
    listener_choice: str
    attempts: int


def default_model(temperature=0) -> ChatOpenAI:
    # Read VLM_* settings from a .env file if present; real environment variables take precedence.
    load_dotenv()
    # llama.cpp serves an OpenAI-compatible API; it ignores the key, so a dummy value is fine.
    
    # retrieve environment variables
    base_url = os.environ.get("VLM_BASE_URL")
    model = os.environ.get("VLM_MODEL")
    api_key = os.environ.get("VLM_API_KEY")  # can be any value for local model
    
    # assert that the relevant environment variables are set
    assert base_url
    assert model
    
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        # Fail fast instead of hanging on an unreachable server; no silent retries.
        timeout=float(os.environ.get("VLM_TIMEOUT", "300")),
        max_retries=int(os.environ.get("VLM_MAX_RETRIES", "0")),
    )


# Node 1: check all three inputs before any model call; raises on the first problem.
def validate(state: State) -> dict:
    for key in IMAGE_KEYS:
        value = state.get(key)
        if value is None or value == "":
            raise ValueError(f"Missing required image input: {key}")
        # PIL images need no further checks.
        if isinstance(value, Image.Image):
            continue
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{key}: expected a file path or PIL.Image.Image, got {type(value).__name__}")
        # Remaining case: a file path; it must exist and be PNG/JPEG.
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{key}: file not found: {value}")
        if path.suffix.lower() not in MIME_TYPES:
            raise ValueError(f"{key}: unsupported image format {path.suffix!r} (use PNG or JPEG)")
    return {}  # validation only; no state update


# Convert one image (PIL object or file path) into an inline base64 data URL for the chat message.
def image_part(image: ImageInput) -> dict:
    if isinstance(image, Image.Image):
        # PIL images are serialized to PNG in memory.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        mime, raw = "image/png", buffer.getvalue()
    else:
        # Files are sent as-is; the MIME type comes from the extension (checked in validate).
        path = Path(image)
        mime, raw = MIME_TYPES[path.suffix.lower()], path.read_bytes()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}


def labeled_images(state: State) -> list[dict]:
    # For each image a text label followed by the image, so roles are unambiguous.
    parts = []
    for key in IMAGE_KEYS:
        parts.append({"type": "text", "text": f"{LABELS[key]}:"})
        parts.append(image_part(state[key]))
    return parts


# rng is injectable so the listener's image order can be made deterministic in tests.
# use_listener=False builds the simple variant: validate -> describe -> END, with no
# listener evaluation and no refinement.
def build_graph(
    model: BaseChatModel | None = None,
    rng: random.Random | None = None,
    use_listener: bool = True,
):
    model = model or default_model()
    rng = rng or random.Random()

    # Node 2: build one multimodal message and make the initial VLM call.
    def describe(state: State) -> dict:
        content = [{"type": "text", "text": INSTRUCTION}] + labeled_images(state)
        # Errors are not caught and reach the caller.
        response = model.invoke([{"role": "user", "content": content}])
        return {"description": response.content}

    # Node 3: a blind listener must pick the described image out of the three, in random order.
    def internal_listener_eval(state: State) -> dict:
        order = list(IMAGE_KEYS)
        rng.shuffle(order)
        # Unlabeled "Image 1..3" so the listener cannot see which one is the target.
        content = [{"type": "text", "text": LISTENER_INSTRUCTION}]
        for number, key in enumerate(order, start=1):
            content.append({"type": "text", "text": f"Image {number}:"})
            content.append(image_part(state[key]))
        content.append({"type": "text", "text": f"Description: {state['description']}"})
        answer = model.invoke([{"role": "user", "content": content}]).content
        match = re.fullmatch(r"(?:Image\s*)?([123])\.?", answer.strip(), re.IGNORECASE)
        if match is None:
            raise ValueError(f"Listener answer is not 1, 2 or 3: {answer!r}")
        # Map the chosen position back to the role of that image.
        return {
            "listener_choice": order[int(match.group(1)) - 1],
            "attempts": state.get("attempts", 0) + 1,
        }

    # Node 4: rewrite the description, telling the model which image the listener confused it with.
    def refine(state: State) -> dict:
        instruction = REFINE_INSTRUCTION.format(chosen=LABELS[state["listener_choice"]])
        content = (
            [{"type": "text", "text": instruction}]
            + labeled_images(state)
            + [{"type": "text", "text": f"Current description: {state['description']}"}]
        )
        response = model.invoke([{"role": "user", "content": content}])
        return {"description": response.content}

    # Finish when the listener found the target; otherwise refine, unless the round limit is hit.
    def route_after_eval(state: State) -> str:
        if state["listener_choice"] == "target":
            return END
        if state["attempts"] >= MAX_EVAL_ROUNDS:
            raise RuntimeError(
                f"No discriminative description after {MAX_EVAL_ROUNDS} rounds; "
                f"last candidate: {state['description']!r}"
            )
        return "refine"

    graph = StateGraph(State)
    graph.add_node("validate", validate)
    graph.add_node("describe", describe)
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "describe")
    if not use_listener:
        # Simple variant: terminate right after the initial description.
        graph.add_edge("describe", END)
        return graph.compile()

    # Full variant: describe -> internal_listener_eval -> (END | refine -> eval ...).
    graph.add_node("internal_listener_eval", internal_listener_eval)
    graph.add_node("refine", refine)
    graph.add_edge("describe", "internal_listener_eval")
    graph.add_conditional_edges("internal_listener_eval", route_after_eval, ["refine", END])
    graph.add_edge("refine", "internal_listener_eval")
    return graph.compile()


# Convenience wrapper: build the graph, run it once and return just the description.
def describe_target(
    target: ImageInput,
    distractor1: ImageInput,
    distractor2: ImageInput,
    model: BaseChatModel | None = None,
    use_listener: bool = True,
) -> str:
    result = build_graph(model, use_listener=use_listener).invoke(
        {"target": target, "distractor1": distractor1, "distractor2": distractor2}
    )
    return result["description"]
