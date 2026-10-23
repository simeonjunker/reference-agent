import base64
import io
import os
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

ImageInput = str | Path | Image.Image


# Graph state: the three image inputs, plus the description written by the "describe" node.
class State(TypedDict, total=False):
    target: ImageInput
    distractor1: ImageInput
    distractor2: ImageInput
    description: str


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


def build_graph(model: BaseChatModel | None = None):
    model = model or default_model()

    # Node 2: build one multimodal message and make the single VLM call.
    def describe(state: State) -> dict:
        # Message layout: instruction, then for each image a text label followed by the image,
        # so the model can tell target and distractors apart by role.
        content = [{"type": "text", "text": INSTRUCTION}]
        for key in IMAGE_KEYS:
            content.append({"type": "text", "text": f"{LABELS[key]}:"})
            content.append(image_part(state[key]))
        # The only model invocation; errors are not caught and reach the caller.
        response = model.invoke([{"role": "user", "content": content}])
        # Write the model's answer to the state unchanged.
        return {"description": response.content}

    # Linear flow: START -> validate -> describe -> END.
    graph = StateGraph(State)
    graph.add_node("validate", validate)
    graph.add_node("describe", describe)
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "describe")
    graph.add_edge("describe", END)
    return graph.compile()


# Convenience wrapper: build the graph, run it once and return just the description.
def describe_target(
    target: ImageInput,
    distractor1: ImageInput,
    distractor2: ImageInput,
    model: BaseChatModel | None = None,
) -> str:
    result = build_graph(model).invoke(
        {"target": target, "distractor1": distractor1, "distractor2": distractor2}
    )
    return result["description"]
