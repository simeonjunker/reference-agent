from pathlib import Path

from PIL import Image, ImageDraw

out = Path(__file__).parent / "images"
out.mkdir(exist_ok=True)

shapes = {
    "target": ("red", "circle"),
    "distractor1": ("blue", "circle"),
    "distractor2": ("red", "square"),
}
for name, (color, shape) in shapes.items():
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    box = (48, 48, 208, 208)
    (draw.ellipse if shape == "circle" else draw.rectangle)(box, fill=color)
    img.save(out / f"{name}.png")
