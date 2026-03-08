from __future__ import annotations

from math import atan2, cos, sin
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont


SCALE = 1.4  # upscale layout for sharper output and larger text


def load_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    """Best-effort font loader with sensible fallback."""
    candidates = []
    if weight == "bold":
        candidates.extend([
            "SegoeUI-Semibold.ttf",
            "Segoe UI Semibold.ttf",
            "Arial Bold.ttf",
        ])
    else:
        candidates.extend([
            "SegoeUI.ttf",
            "Segoe UI.ttf",
            "Arial.ttf",
        ])

    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    lines: Iterable[str],
    font: ImageFont.FreeTypeFont,
    fill: str,
    line_height: int,
):
    x0, y0, x1, y1 = box
    text_lines = list(lines)
    total_h = line_height * len(text_lines)
    start_y = y0 + (y1 - y0 - total_h) // 2
    for i, text in enumerate(text_lines):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        x = x0 + (x1 - x0 - w) // 2
        y = start_y + i * line_height
        draw.text((x, y), text, font=font, fill=fill)


def rounded(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], radius: int, fill: str, outline: str, width: int = 2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color: str, width: int = 5, head: int = 22):
    draw.line([start, end], fill=color, width=width)
    angle = atan2(end[1] - start[1], end[0] - start[0])
    left = (end[0] - head * cos(angle - 0.35), end[1] - head * sin(angle - 0.35))
    right = (end[0] - head * cos(angle + 0.35), end[1] - head * sin(angle + 0.35))
    draw.polygon([end, left, right], fill=color)


def scale_box(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return tuple(int(v * SCALE) for v in box)


def scale_point(pt: Tuple[int, int]) -> Tuple[int, int]:
    return (int(pt[0] * SCALE), int(pt[1] * SCALE))


def flowchart_png(out_path: Path):
    bg = "#f9fbfd"
    blue = "#12406a"
    green = "#2e7d32"
    clay = "#8d6e63"

    img = Image.new("RGB", (int(1100 * SCALE), int(720 * SCALE)), color=bg)
    draw = ImageDraw.Draw(img)

    font_title = load_font(int(18 * SCALE), "bold")
    font_body = load_font(int(15 * SCALE), "regular")

    # Boxes
    boxes = {
        "camera": (scale_box((420, 50, 680, 120)), ["Camera frame"], "#ffffff", blue, font_title),
        "yolo": (scale_box((420, 150, 680, 240)), ["YOLO ROI extraction", "detect feces, crop region"], "#ffffff", blue, font_title),
        "edge": (scale_box((420, 270, 680, 340)), ["EfficientNetB0 edge screening"], "#ffffff", blue, font_title),
        "gate": (scale_box((410, 370, 690, 440)), ["Safety / confidence gate"], "#eef7f0", green, font_title),
        "healthy": (scale_box((140, 470, 400, 560)), ["Healthy pipeline", "safe & confident"], "#ffffff", blue, font_title),
        "cloud_ver": (scale_box((700, 470, 960, 540)), ["Cloud verification"], "#ffffff", blue, font_title),
        "ensemble": (scale_box((670, 560, 990, 650)), ["Cloud ensemble", "EfficientNetB0 + DenseNet121 + safety vote"], "#fdf4ec", clay, font_title),
        "final": (scale_box((420, 580, 680, 660)), ["Final label + isolate/alert decision"], "#ffffff", blue, font_title),
    }

    for key, (xy, lines, fill, outline, font) in boxes.items():
        rounded(draw, xy, radius=16, fill=fill, outline=outline, width=3 if key in {"gate", "ensemble"} else 2)
        draw_centered_text(draw, xy, lines, font=font, fill=outline, line_height=int(26 * SCALE))

    # Main arrows
    draw_arrow(draw, scale_point((550, 120)), scale_point((550, 150)), blue)
    draw_arrow(draw, scale_point((550, 240)), scale_point((550, 270)), blue)
    draw_arrow(draw, scale_point((550, 340)), scale_point((550, 370)), blue)

    # Branch arrows
    draw_arrow(draw, scale_point((550, 440)), scale_point((340, 485)), blue)
    draw_arrow(draw, scale_point((550, 440)), scale_point((760, 490)), blue)
    draw_arrow(draw, scale_point((830, 540)), scale_point((830, 560)), clay)

    # Merge arrows
    draw_arrow(draw, scale_point((330, 545)), scale_point((520, 590)), blue)
    draw_arrow(draw, scale_point((830, 650)), scale_point((590, 620)), clay)

    img = img.resize((int(img.width / (SCALE / 1.05)), int(img.height / (SCALE / 1.05))), resample=Image.Resampling.LANCZOS)
    img.save(out_path, format="PNG")


def architecture_png(out_path: Path):
    bg = "#f8fbff"
    blue = "#12406a"
    clay = "#8d6e63"

    img = Image.new("RGB", (int(1200 * SCALE), int(720 * SCALE)), color=bg)
    draw = ImageDraw.Draw(img)

    font_title = load_font(int(18 * SCALE), "bold")
    font_body = load_font(int(15 * SCALE), "regular")

    # Frames
    frame_edge = scale_box((60, 70, 520, 630))
    frame_cloud = scale_box((620, 60, 1140, 630))
    rounded(draw, frame_edge, radius=18, fill="#ffffff", outline=blue, width=3)
    rounded(draw, frame_cloud, radius=18, fill="#ffffff", outline=clay, width=3)
    draw_centered_text(draw, frame_edge, ["Edge device (farm site)"], font_title, blue, int(26 * SCALE))
    draw_centered_text(draw, frame_cloud, ["Cloud services"], font_title, clay, int(26 * SCALE))

    # Edge boxes
    edge_boxes = [
        (scale_box((100, 140, 480, 210)), ["Camera sensor (conveyor feed)"]),
        (scale_box((100, 230, 480, 320)), ["YOLO ROI extraction", "detect feces, crop region"]),
        (scale_box((100, 340, 480, 420)), ["EfficientNetB0 (TFLite)", "fast edge screening"]),
        (scale_box((100, 440, 480, 520)), ["Safety / confidence gate", "routes uncertain cases"]),
        (scale_box((140, 540, 440, 610)), ["Edge actions", "pass, isolate, or escalate"]),
    ]
    for xy, lines in edge_boxes:
        rounded(draw, xy, radius=14, fill="#f5f8fc", outline=blue, width=2)
        draw_centered_text(draw, xy, lines, font_title, blue, int(26 * SCALE))

    # Cloud boxes
    cloud_boxes = [
        (scale_box((660, 130, 1100, 210)), ["Secure ingest API", "receives escalated ROIs"]),
        (scale_box((660, 230, 1100, 340)), ["Ensemble inference", "EfficientNetB0 + DenseNet121", "safety vote & uncertainty check"]),
        (scale_box((660, 360, 1100, 450)), ["Decisioning + alerting", "final label, isolate/alert decision"]),
        (scale_box((660, 470, 870, 550)), ["Model registry", "checkpoints, tflite/onnx"]),
        (scale_box((910, 470, 1100, 550)), ["Observability", "logs, metrics, audits"]),
        (scale_box((710, 570, 1050, 640)), ["Operator dashboard / notifications"]),
    ]
    for xy, lines in cloud_boxes:
        rounded(draw, xy, radius=14, fill="#fdf5ee", outline=clay, width=2)
        draw_centered_text(draw, xy, lines, font_title, clay, int(26 * SCALE))

    # Edge arrows down
    draw_arrow(draw, scale_point((290, 210)), scale_point((290, 230)), blue)
    draw_arrow(draw, scale_point((290, 320)), scale_point((290, 340)), blue)
    draw_arrow(draw, scale_point((290, 420)), scale_point((290, 440)), blue)
    draw_arrow(draw, scale_point((290, 520)), scale_point((290, 540)), blue)

    # Edge -> cloud escalation
    draw_arrow(draw, scale_point((480, 500)), scale_point((770, 190)), blue)
    draw.text(scale_point((520, 360)), "uncertain / suspicious", font=font_body, fill=blue)
    draw.text(scale_point((170, 330)), "safe & confident", font=font_body, fill=blue)

    # Cloud arrows down
    draw_arrow(draw, scale_point((880, 210)), scale_point((880, 230)), clay)
    draw_arrow(draw, scale_point((880, 340)), scale_point((880, 360)), clay)
    draw_arrow(draw, scale_point((880, 450)), scale_point((880, 470)), clay)
    draw_arrow(draw, scale_point((765, 550)), scale_point((765, 570)), clay)
    draw_arrow(draw, scale_point((1005, 550)), scale_point((1005, 570)), clay)

    # Return path
    draw_arrow(draw, scale_point((880, 640)), scale_point((380, 620)), clay)
    draw.text(scale_point((560, 660)), "return label / alert to edge", font=font_body, fill=clay)

    img = img.resize((int(img.width / (SCALE / 1.05)), int(img.height / (SCALE / 1.05))), resample=Image.Resampling.LANCZOS)
    img.save(out_path, format="PNG")


def main():
    base = Path(__file__).parent
    flowchart_png(base / "system_flowchart.png")
    architecture_png(base / "architecture_overview.png")


if __name__ == "__main__":
    main()