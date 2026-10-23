import base64
import io

import pytest
from PIL import Image
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from reference_agent.graph import INSTRUCTION, build_graph, describe_target


# Fake model that returns a canned answer and records every message list it receives,
# so tests can inspect exactly what the graph sent to the model.
class RecordingModel(GenericFakeChatModel):
    calls: list = []

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return super()._generate(messages, *args, **kwargs)


def make_model(text="A red car."):
    return RecordingModel(messages=iter([AIMessage(content=text)]), calls=[])


# Three tiny PNG files with distinct contents (b"T", b"D1", b"D2"), so a swapped role is detectable.
@pytest.fixture
def images(tmp_path):
    paths = {}
    for name, data in {"target": b"T", "distractor1": b"D1", "distractor2": b"D2"}.items():
        p = tmp_path / f"{name}.png"
        p.write_bytes(data)
        paths[name] = str(p)
    return paths


def test_returns_description_unchanged(images):
    """The model's answer ends up in result["description"] without modification."""
    result = build_graph(make_model("A red car.")).invoke(images)
    assert result["description"] == "A red car."


def test_model_receives_instruction_and_images_in_order(images):
    """The model is called exactly once with the instruction, followed by the labelled images
    in the order target, distractor 1, distractor 2, each with its own file content."""
    model = make_model()
    build_graph(model).invoke(images)

    assert len(model.calls) == 1
    (message,) = model.calls[0]
    parts = message.content
    assert parts[0] == {"type": "text", "text": INSTRUCTION}

    decoded = []
    for label_part, image in zip(parts[1::2], parts[2::2]):
        assert image["type"] == "image_url"
        url = image["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded.append((label_part["text"], base64.b64decode(url.split(",", 1)[1])))
    assert decoded == [("Target:", b"T"), ("Distractor 1:", b"D1"), ("Distractor 2:", b"D2")]


@pytest.mark.parametrize("missing", ["target", "distractor1", "distractor2"])
def test_missing_input_rejected(images, missing):
    """Omitting any one of the three required inputs raises a ValueError naming it."""
    del images[missing]
    with pytest.raises(ValueError, match=missing):
        build_graph(make_model()).invoke(images)


def test_nonexistent_file_rejected(images):
    """A path that does not exist raises FileNotFoundError naming the offending input."""
    images["distractor1"] = "/nonexistent/x.png"
    with pytest.raises(FileNotFoundError, match="distractor1"):
        build_graph(make_model()).invoke(images)


def test_unsupported_format_rejected(images, tmp_path):
    """A file with an unsupported extension (GIF) is rejected with a ValueError."""
    bad = tmp_path / "x.gif"
    bad.write_bytes(b"G")
    images["distractor2"] = str(bad)
    with pytest.raises(ValueError, match="unsupported"):
        build_graph(make_model()).invoke(images)


def test_invalid_input_does_not_call_model(images):
    """Validation fails before the model is invoked, so invalid input never reaches the model."""
    model = make_model()
    del images["target"]
    with pytest.raises(ValueError):
        build_graph(model).invoke(images)
    assert model.calls == []


def test_model_failure_propagates(images):
    """An exception raised by the model is not swallowed and reaches the caller of invoke()."""
    class Failing(GenericFakeChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("server down")

    with pytest.raises(RuntimeError, match="server down"):
        build_graph(Failing(messages=iter([]))).invoke(images)


def test_pil_images_encoded_and_roles_preserved(images):
    """PIL images are encoded as PNG and mixing them with file paths keeps each image in its role
    (checked via pixel colour for PIL inputs and file content for the path input)."""
    colors = {"target": "red", "distractor1": "green", "distractor2": "blue"}
    model = make_model()
    inputs = {k: Image.new("RGB", (4, 4), c) for k, c in colors.items()}
    inputs["distractor1"] = images["distractor1"]  # mixed path and PIL input
    build_graph(model).invoke(inputs)

    parts = model.calls[0][0].content
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    first = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1])))
    assert first.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert base64.b64decode(urls[1].split(",", 1)[1]) == b"D1"
    last = Image.open(io.BytesIO(base64.b64decode(urls[2].split(",", 1)[1])))
    assert last.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_unsupported_input_type_rejected(images):
    """An input that is neither a path nor a PIL image (here an int) raises a TypeError."""
    images["target"] = 123
    with pytest.raises(TypeError, match="target"):
        build_graph(make_model()).invoke(images)


def test_describe_target_returns_str(images):
    """The convenience wrapper returns the description as a plain string."""
    out = describe_target(images["target"], images["distractor1"], images["distractor2"], make_model("A cat."))
    assert out == "A cat."
