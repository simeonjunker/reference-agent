import base64
import io

import pytest
from PIL import Image
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from reference_agent import graph as graph_module
from reference_agent.graph import INSTRUCTION, build_graph, describe_target


# Fake model that returns a canned answer and records every message list it receives,
# so tests can inspect exactly what the graph sent to the model.
class RecordingModel(GenericFakeChatModel):
    calls: list = []

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return super()._generate(messages, *args, **kwargs)


# Deterministic stand-in for random.Random: the listener sees the images reversed, i.e.
# Image 1 = distractor2, Image 2 = distractor1, Image 3 = target.
class ReverseRng:
    def shuffle(self, items):
        items.reverse()


LISTENER_PICKS_TARGET = "3"
LISTENER_PICKS_DISTRACTOR1 = "2"
LISTENER_PICKS_DISTRACTOR2 = "1"


# Scripted model: replies are returned in order, one per model call.
def make_model(*replies):
    replies = replies or ("A red car.", LISTENER_PICKS_TARGET)
    return RecordingModel(messages=iter([AIMessage(content=r) for r in replies]), calls=[])


def graph_for(model):
    return build_graph(model, rng=ReverseRng())


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
    result = graph_for(make_model("A red car.", LISTENER_PICKS_TARGET)).invoke(images)
    assert result["description"] == "A red car."


def test_model_receives_instruction_and_images_in_order(images):
    """The first model call carries the instruction, followed by the labelled images
    in the order target, distractor 1, distractor 2, each with its own file content."""
    model = make_model()
    graph_for(model).invoke(images)

    assert len(model.calls) == 2  # describe + listener eval
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
        graph_for(make_model()).invoke(images)


def test_nonexistent_file_rejected(images):
    """A path that does not exist raises FileNotFoundError naming the offending input."""
    images["distractor1"] = "/nonexistent/x.png"
    with pytest.raises(FileNotFoundError, match="distractor1"):
        graph_for(make_model()).invoke(images)


def test_unsupported_format_rejected(images, tmp_path):
    """A file with an unsupported extension (GIF) is rejected with a ValueError."""
    bad = tmp_path / "x.gif"
    bad.write_bytes(b"G")
    images["distractor2"] = str(bad)
    with pytest.raises(ValueError, match="unsupported"):
        graph_for(make_model()).invoke(images)


def test_invalid_input_does_not_call_model(images):
    """Validation fails before the model is invoked, so invalid input never reaches the model."""
    model = make_model()
    del images["target"]
    with pytest.raises(ValueError):
        graph_for(model).invoke(images)
    assert model.calls == []


def test_model_failure_propagates(images):
    """An exception raised by the model is not swallowed and reaches the caller of invoke()."""
    class Failing(GenericFakeChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("server down")

    with pytest.raises(RuntimeError, match="server down"):
        graph_for(Failing(messages=iter([]))).invoke(images)


def test_pil_images_encoded_and_roles_preserved(images):
    """PIL images are encoded as PNG and mixing them with file paths keeps each image in its role
    (checked via pixel colour for PIL inputs and file content for the path input)."""
    colors = {"target": "red", "distractor1": "green", "distractor2": "blue"}
    model = make_model()
    inputs = {k: Image.new("RGB", (4, 4), c) for k, c in colors.items()}
    inputs["distractor1"] = images["distractor1"]  # mixed path and PIL input
    graph_for(model).invoke(inputs)

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
        graph_for(make_model()).invoke(images)


def test_describe_target_returns_str(images, monkeypatch):
    """The convenience wrapper returns the description as a plain string."""
    monkeypatch.setattr(graph_module.random, "Random", ReverseRng)  # wrapper has no rng argument
    out = describe_target(
        images["target"], images["distractor1"], images["distractor2"], make_model("A cat.", LISTENER_PICKS_TARGET)
    )
    assert out == "A cat."


def image_urls(call):
    return [p["image_url"]["url"] for p in call[0].content if p["type"] == "image_url"]


def payload(url):
    return base64.b64decode(url.split(",", 1)[1])


def test_accepted_description_skips_refine(images):
    """If the listener picks the target, the graph stops after one evaluation without refining."""
    model = make_model("A red car.", LISTENER_PICKS_TARGET)
    result = graph_for(model).invoke(images)
    assert result["description"] == "A red car."
    assert result["attempts"] == 1
    assert len(model.calls) == 2


def test_listener_sees_unlabeled_shuffled_images_and_description(images):
    """The listener gets the three images in shuffled order, without role labels, plus the description."""
    model = make_model("A red car.", LISTENER_PICKS_TARGET)
    graph_for(model).invoke(images)
    listener_call = model.calls[1]
    assert [payload(u) for u in image_urls(listener_call)] == [b"D2", b"D1", b"T"]
    texts = [p["text"] for p in listener_call[0].content if p["type"] == "text"]
    assert not any("Target" in t or "Distractor" in t for t in texts)
    assert "Description: A red car." in texts


def test_wrong_pick_triggers_refine_then_reevaluation(images):
    """A wrong pick leads to refine, then eval again; the refined description is returned."""
    model = make_model("vague", LISTENER_PICKS_DISTRACTOR1, "a red car", LISTENER_PICKS_TARGET)
    result = graph_for(model).invoke(images)
    assert result["description"] == "a red car"
    assert result["attempts"] == 2
    assert len(model.calls) == 4


def test_refine_receives_labeled_images_old_description_and_confused_role(images):
    """Refine sees the images with their true roles, the current description and the mistaken role."""
    model = make_model("vague", LISTENER_PICKS_DISTRACTOR2, "better", LISTENER_PICKS_TARGET)
    graph_for(model).invoke(images)
    refine_call = model.calls[2]
    assert [payload(u) for u in image_urls(refine_call)] == [b"T", b"D1", b"D2"]
    texts = [p["text"] for p in refine_call[0].content if p["type"] == "text"]
    assert "Current description: vague" in texts
    assert any("Distractor 2" in t and "wrongly" in t for t in texts)


def test_loop_stops_with_error_after_max_rounds(images, monkeypatch):
    """If the listener never finds the target, the loop ends with an error at the round limit."""
    monkeypatch.setattr(graph_module, "MAX_EVAL_ROUNDS", 3)
    # 1 describe, then (eval, refine) pairs; the third eval fails and raises.
    model = make_model("d0", "1", "d1", "1", "d2", "1")
    with pytest.raises(RuntimeError, match="3 rounds"):
        graph_for(model).invoke(images)
    assert len(model.calls) == 6


def test_unparsable_listener_answer_rejected(images):
    """A listener answer that is not 1, 2 or 3 raises a ValueError instead of being guessed."""
    with pytest.raises(ValueError, match="Listener answer"):
        graph_for(make_model("A red car.", "probably the red one")).invoke(images)


def test_variant_without_listener_terminates_after_describe(images):
    """With use_listener=False the graph makes a single model call and returns the first description."""
    model = make_model("A red car.")
    result = build_graph(model, use_listener=False).invoke(images)
    assert result["description"] == "A red car."
    assert len(model.calls) == 1
    assert "attempts" not in result and "listener_choice" not in result


def test_variant_without_listener_has_no_loop_nodes():
    """The simple variant contains only validate and describe."""
    g = build_graph(make_model(), use_listener=False).get_graph()
    assert set(g.nodes) == {"__start__", "validate", "describe", "__end__"}


def test_describe_target_without_listener(images):
    """The wrapper forwards use_listener; no listener reply is needed."""
    out = describe_target(
        images["target"], images["distractor1"], images["distractor2"], make_model("A cat."), use_listener=False
    )
    assert out == "A cat."
