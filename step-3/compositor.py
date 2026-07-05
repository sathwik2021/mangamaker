# compositor.py
#
# Step 3e: Layout JSON + panel images → full manga page
#
# Usage:
#   python compositor.py \
#     --layout   page_abc123.json \
#     --panels   panels/page_abc123/ \
#     --output   output/
#
# Batch:
#   python compositor.py \
#     --layout_dir  step-2-layout/output/ \
#     --panels_dir  step-3/panels/ \
#     --output      step-3/output/
#     --workers     4

import argparse
import json
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch

# ── Optional: TextRenderer for advanced text rendering ─────────────────────
try:
    from text_renderer import TextRenderer
    TEXT_RENDERER_AVAILABLE = True
except ImportError:
    TEXT_RENDERER_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compositor")

# ── Global text-measurement surface (reused, never drawn to canvas) ──────────
_MEASURE_IMG  = Image.new("L", (1, 1))
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMG)

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ── Canvas defaults ───────────────────────────────────────────────────────────
CANVAS_W  = 1800
CANVAS_H  = 2400
CANVAS_BG = (255, 255, 255)

# ── Panel borders ─────────────────────────────────────────────────────────────
BORDER_COLOR = (0, 0, 0)
BORDER_WIDTH = 6

# ── Speech bubbles ─────────────────────────────────────────────────────────────
BUBBLE_FILL     = (255, 255, 255)
BUBBLE_STROKE   = (0, 0, 0)
BUBBLE_STROKE_W = 3
BUBBLE_PAD      = 18
TAIL_LENGTH     = 48

# ── Typography ─────────────────────────────────────────────────────────────────
FONT_SIZE_DEFAULT   = 28
FONT_SIZE_WHISPER   = 22
FONT_SIZE_SHOUT     = 34
FONT_SIZE_NARRATION = 24
LINE_SPACING        = 6

# Manga / comic fonts searched in priority order; fall back to system sans-serif
FONT_PATHS: List[str] = [
    # --- Manga / comic fonts (common free installs) ---
    "/usr/share/fonts/truetype/bangers/Bangers-Regular.ttf",
    "/usr/share/fonts/truetype/comic-relief/ComicRelief-Regular.ttf",
    "/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf",
    "/usr/share/fonts/OTF/Komika_Text.otf",
    "/usr/share/fonts/truetype/Komika_Text.ttf",
    # macOS comic-ish
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/Library/Fonts/Comic Sans MS.ttf",
    # Windows comic fonts
    "C:/Windows/Fonts/comic.ttf",
    "C:/Windows/Fonts/comicbd.ttf",
    "C:/Windows/Fonts/Bangers-Regular.ttf",
    # Generic sans-serif fallbacks
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


@lru_cache(maxsize=32)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load and cache the best available font at the given size."""
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    logger.warning("No TrueType font found — using PIL default (quality will be poor)")
    return ImageFont.load_default()


def _font_for_bubble(bubble_type: str) -> ImageFont.FreeTypeFont:
    sizes = {
        "whisper":   FONT_SIZE_WHISPER,
        "shout":     FONT_SIZE_SHOUT,
        "narration": FONT_SIZE_NARRATION,
    }
    return _load_font(sizes.get(bubble_type, FONT_SIZE_DEFAULT))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class CompositorConfig:
    """Central, serialisable configuration for the compositor."""
    canvas_w: int = CANVAS_W
    canvas_h: int = CANVAS_H
    canvas_bg: Tuple[int, int, int] = (255, 255, 255)
    border_color: Tuple[int, int, int] = (0, 0, 0)
    border_width: int = BORDER_WIDTH
    bubble_fill: Tuple[int, int, int] = BUBBLE_FILL
    bubble_stroke: Tuple[int, int, int] = BUBBLE_STROKE
    bubble_stroke_w: int = BUBBLE_STROKE_W
    bubble_pad: int = BUBBLE_PAD
    tail_length: int = TAIL_LENGTH
    font_size_default: int = FONT_SIZE_DEFAULT
    font_size_whisper: int = FONT_SIZE_WHISPER
    font_size_shout: int = FONT_SIZE_SHOUT
    font_size_narration: int = FONT_SIZE_NARRATION
    line_spacing: int = LINE_SPACING
    save_panels: bool = True
    validate: bool = True
    organic_wobble: int = 6        # px jitter for hand-drawn ellipse effect
    shout_spike_count: int = 20    # number of spikes on explosion bubbles
    shout_spike_depth: float = 0.28  # relative depth of shout spikes


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _expand_bbox(
    bbox: List[int],
    tw: int,
    th: int,
    pad: int,
) -> List[int]:
    """Expand a bbox so that a text block of (tw, th) fits with padding."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ex1 = int(min(x1, cx - tw / 2 - pad))
    ex2 = int(max(x2, cx + tw / 2 + pad))
    ey1 = int(min(y1, cy - th / 2 - pad))
    ey2 = int(max(y2, cy + th / 2 + pad))
    return [ex1, ey1, ex2, ey2]


# ── Text helpers ───────────────────────────────────────────────────────────────

def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    """Word-wrap, respecting embedded newlines."""
    if not text:
        return []
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if _MEASURE_DRAW.textlength(test, font=font) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _fit_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_height: int,
) -> Tuple[List[str], ImageFont.FreeTypeFont]:
    """
    Shrink font until wrapped text fits within max_width × max_height.
    Returns (lines, fitted_font).
    """
    size = font.size
    while size > 10:
        f     = _load_font(size)
        lines = _wrap_text(text, f, max_width)
        _, h  = _text_block_size(lines, f)
        if h <= max_height:
            return lines, f
        size -= 2
    f     = _load_font(10)
    lines = _wrap_text(text, f, max_width)
    return lines, f


def _text_block_size(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: int = LINE_SPACING,
) -> Tuple[int, int]:
    lh = font.size + line_spacing
    w  = max(
        (int(_MEASURE_DRAW.textlength(l, font=font)) for l in lines),
        default=0,
    )
    h  = lh * len(lines)
    return w, h


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    cx: float,
    cy: float,
    th: int,
    fill: Tuple[int, int, int] = (0, 0, 0),
    stroke_fill: Optional[Tuple[int, int, int]] = None,
    stroke_width: int = 0,
    line_spacing: int = LINE_SPACING,
) -> None:
    lh      = font.size + line_spacing
    y_start = int(cy - th / 2)
    for i, line in enumerate(lines):
        lw = int(_MEASURE_DRAW.textlength(line, font=font))
        lx = int(cx - lw / 2)
        ly = y_start + i * lh
        if stroke_fill and stroke_width:
            draw.text((lx, ly), line, font=font,
                      fill=fill, stroke_width=stroke_width,
                      stroke_fill=stroke_fill)
        else:
            draw.text((lx, ly), line, fill=fill, font=font)


# ── Organic / hand-drawn polygon helpers ────────────────────────────────────

def _jittered_ellipse_poly(
    cx: float, cy: float,
    rx: float, ry: float,
    steps: int = 52,
    wobble: float = 3.2,
) -> List[Tuple[float, float]]:
    """
    Return a polygon that approximates an ellipse with subtle jitter on
    every vertex, giving a hand-drawn / organic look.
    """
    pts = []
    for i in range(steps):
        angle = 2 * math.pi * i / steps
        jx    = random.uniform(-wobble, wobble)
        jy    = random.uniform(-wobble, wobble)
        x     = cx + (rx + jx) * math.cos(angle)
        y     = cy + (ry + jy) * math.sin(angle)
        pts.append((x, y))
    return pts


def _jittered_line_poly(
    points: List[Tuple[float, float]],
    wobble: float = 1.2,
) -> List[Tuple[float, float]]:
    """
    Return a polygon with jittered vertices to give a hand-drawn line effect.
    Adds random offset to each point without changing the overall structure.
    """
    jittered = []
    for x, y in points:
        jx = random.uniform(-wobble, wobble)
        jy = random.uniform(-wobble, wobble)
        jittered.append((x + jx, y + jy))
    return jittered


def _shout_explosion_poly(
    cx: float,
    cy: float,
    inner_rx: float,
    inner_ry: float,
    outer_rx: float,
    outer_ry: float,
    spikes: int = 20,
) -> List[Tuple[int, int]]:
    """
    Return a star/explosion polygon for shout bubbles.
    Alternates between inner ellipse and outer ellipse vertices.
    """
    pts = []
    for i in range(spikes * 2):
        angle = 2 * math.pi * i / (spikes * 2)
        if i % 2 == 0:
            # outer spike
            rx_ = outer_rx + random.uniform(-4, 4)
            ry_ = outer_ry + random.uniform(-4, 4)
        else:
            # inner valley
            rx_ = inner_rx + random.uniform(-3, 3)
            ry_ = inner_ry + random.uniform(-3, 3)
        pts.append((int(cx + rx_ * math.cos(angle)),
                    int(cy + ry_ * math.sin(angle))))
    return pts


def _cloud_puff_poly(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    puffs: int = 12,
) -> List[Tuple[int, int]]:
    """
    Return a cloud/puff polygon for thought bubbles.
    Creates organic bumpy edges by adding circular puffs around an ellipse.
    """
    pts = []
    puff_radius = max(rx, ry) * 0.3  # Size of each puff
    
    for i in range(puffs * 4):  # More vertices for smoother shape
        angle = 2 * math.pi * i / (puffs * 4)
        
        # Base ellipse point
        base_x = cx + rx * math.cos(angle)
        base_y = cy + ry * math.sin(angle)
        
        # Add puff perturbation
        puff_idx = i // 4
        puff_angle = 2 * math.pi * puff_idx / puffs
        puff_offset = puff_radius * 0.7 * (1 + 0.3 * math.sin(angle * puffs))
        
        x = base_x + puff_offset * math.cos(puff_angle)
        y = base_y + puff_offset * math.sin(puff_angle)
        
        pts.append((int(x), int(y)))
    
    return pts


def _tapered_tail_poly(
    bx: float, by: float,
    tx: float, ty: float,
    base_half: float = 12.0,
) -> List[Tuple[int, int]]:
    """
    Return a tapered (triangular) tail polygon from bubble edge (bx,by)
    to tip (tx,ty).  The base is centred on (bx,by) and perpendicular to
    the tail direction.
    """
    dx   = tx - bx
    dy   = ty - by
    dist = math.hypot(dx, dy) or 1.0
    # Unit perpendicular
    px   = -dy / dist
    py   =  dx / dist
    p1   = (int(bx + px * base_half), int(by + py * base_half))
    p2   = (int(bx - px * base_half), int(by - py * base_half))
    tip  = (int(tx), int(ty))
    return [p1, p2, tip]


# ── Bubble-edge anchor ─────────────────────────────────────────────────────────

def _ellipse_edge_point(
    cx: float, cy: float,
    rx: float, ry: float,
    target_x: float, target_y: float,
) -> Tuple[float, float]:
    """Return the point on the ellipse closest to (target_x, target_y)."""
    angle  = math.atan2(target_y - cy, target_x - cx)
    ex     = cx + rx * math.cos(angle)
    ey     = cy + ry * math.sin(angle)
    return ex, ey


# ── Bubble renderers ──────────────────────────────────────────────────────────

def _draw_ellipse_bubble(
    draw: ImageDraw.ImageDraw,
    bbox: List[int],
    text: str,
    bubble_type: str,
    panel_bbox: Optional[List[int]] = None,
    cfg: CompositorConfig = CompositorConfig(),
) -> None:
    x1, y1, x2, y2 = bbox
    cx, cy = _center(bbox)
    font   = _font_for_bubble(bubble_type)

    bw       = x2 - x1 - cfg.bubble_pad * 2
    bh       = y2 - y1 - cfg.bubble_pad * 2
    lines, font = _fit_text(text, font, max(bw, 60), max(bh, 40))
    tw, th   = _text_block_size(lines, font)

    ebbox    = _expand_bbox(bbox, tw, th, cfg.bubble_pad)
    ex1, ey1, ex2, ey2 = ebbox
    cx, cy   = (ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0
    rx, ry   = (ex2 - ex1) / 2.0, (ey2 - ey1) / 2.0

    # ── Tail ──────────────────────────────────────────────────────────────────
    if panel_bbox and bubble_type not in ["narration", "thought"]:
        pcx, pcy   = _center(panel_bbox)
        ex, ey     = _ellipse_edge_point(cx, cy, rx, ry, pcx, pcy)
        angle      = math.atan2(pcy - cy, pcx - cx)
        tip_x      = cx + math.cos(angle) * (rx + cfg.tail_length)
        tip_y      = cy + math.sin(angle) * (ry + cfg.tail_length)
        tail_pts   = _tapered_tail_poly(ex, ey, tip_x, tip_y, base_half=11)
        draw.polygon(tail_pts, fill=cfg.bubble_fill, outline=cfg.bubble_stroke)
        draw.polygon(tail_pts, fill=cfg.bubble_fill) # Clean seam

    # ── Organic bubble body ─────────────────────────────────────────────
    if bubble_type == "shout":
        inner_rx, inner_ry = rx * 0.72, ry * 0.72
        outer_rx_ = rx + rx * cfg.shout_spike_depth
        outer_ry_ = ry + ry * cfg.shout_spike_depth
        poly = _shout_explosion_poly(cx, cy, inner_rx, inner_ry, outer_rx_, outer_ry_, spikes=cfg.shout_spike_count)
        draw.polygon(poly, fill=cfg.bubble_fill, outline=cfg.bubble_stroke)
    elif bubble_type == "thought":
        poly = _cloud_puff_poly(cx, cy, rx, ry, puffs=12)
        draw.polygon(poly, fill=cfg.bubble_fill, outline=cfg.bubble_stroke)
        # Add a couple of "thought bubbles" trailing off if possible
        # (This would need panel_bbox to avoid clipping, skipping for now)
    else:
        # Organic hand-drawn ellipse (two slightly offset strokes)
        poly = _jittered_ellipse_poly(cx, cy, rx, ry, steps=52, wobble=cfg.organic_wobble)
        draw.polygon(poly, fill=cfg.bubble_fill, outline=None)
        
        # Outer stroke — a second jittered pass for character
        poly2 = _jittered_ellipse_poly(cx, cy, rx + 1, ry + 1, steps=52, wobble=max(1, cfg.organic_wobble // 2))
        draw.line(poly2 + [poly2[0]], fill=cfg.bubble_stroke, width=cfg.bubble_stroke_w)

        if bubble_type == "whisper":
            dash_poly = _jittered_ellipse_poly(cx, cy, rx - 6, ry - 6, steps=52, wobble=2)
            for i in range(0, len(dash_poly) - 1, 2):
                draw.line([dash_poly[i], dash_poly[i + 1]], fill=cfg.bubble_stroke, width=1)

    # ── Render text ───────────────────────────────────────────────────────────
    stroke_kw: Dict[str, Any] = {}
    if bubble_type == "shout":
        stroke_kw = {"stroke_fill": (0, 0, 0), "stroke_width": 1}
    _draw_text_block(draw, lines, font, cx, cy, th, **stroke_kw)


def _draw_narration_box(
    draw: ImageDraw.ImageDraw,
    bbox: List[int],
    text: str,
    cfg: CompositorConfig = CompositorConfig(),
) -> None:
    x1, y1, x2, y2 = bbox
    font  = _load_font(cfg.font_size_narration)
    lines, font = _fit_text(
        text, font,
        x2 - x1 - cfg.bubble_pad * 2,
        y2 - y1 - cfg.bubble_pad * 2,
    )
    draw.rectangle([x1, y1, x2, y2],
                   fill=(235, 235, 235),
                   outline=cfg.bubble_stroke,
                   width=cfg.bubble_stroke_w)
    # Inner rule for visual polish
    draw.rectangle([x1 + 4, y1 + 4, x2 - 4, y2 - 4],
                   outline=(180, 180, 180), width=1)
    cx, cy = _center(bbox)
    _, th  = _text_block_size(lines, font)
    _draw_text_block(draw, lines, font, cx, cy, th)


# ── Panel image helpers ────────────────────────────────────────────────────────

def _safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as exc:
        logger.warning("Could not open %s: %s", path, exc)
        return None


def _find_panel_image(
    panels_dir: Optional[Path],
    page_id: str,
    panel_idx: int,
) -> Optional[Image.Image]:
    if panels_dir is None:
        return None
    patterns = [
        f"panel_{panel_idx}.png",
        f"panel_{panel_idx:03d}.png",
        f"p{panel_idx}.png",
        f"{page_id}_panel_{panel_idx}.png",
        f"{page_id}_panel_{panel_idx:03d}.png",
        f"page_{page_id}_panel_{panel_idx}.png",
    ]
    for pattern in patterns:
        for base in (panels_dir, panels_dir / page_id):
            candidate = base / pattern
            if candidate.exists():
                return _safe_open_image(candidate)
    return None


def _resize_with_aspect(
    img: Image.Image,
    target_w: int,
    target_h: int,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Resize preserving aspect ratio; letterbox with bg colour."""
    img_copy = img.copy()
    img_copy.thumbnail((target_w, target_h), Image.LANCZOS)
    result = Image.new("RGB", (target_w, target_h), bg)
    offset = ((target_w - img_copy.width) // 2,
              (target_h - img_copy.height) // 2)
    result.paste(img_copy, offset)
    return result


# ── Panel placement ────────────────────────────────────────────────────────────

def _place_panel(
    canvas: Image.Image,
    panel_img: Optional[Image.Image],
    bbox: List[int],
    cfg: CompositorConfig = CompositorConfig(),
    clip_polygon: Optional[List[List[float]]] = None,
    bleed: bool = False,
) -> None:
    """Resizes and pastes a panel image into the bbox, with support for clipping and bleeds."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    # Handle bleed — extend to canvas edges if requested
    if bleed:
        if x1 < 25: x1 = 0
        if y1 < 25: y1 = 0
        if x2 > cfg.canvas_w - 25: x2 = cfg.canvas_w
        if y2 > cfg.canvas_h - 25: y2 = cfg.canvas_h
        w, h = x2 - x1, y2 - y1

    if panel_img is not None:
        # Aspect-correct resize & center-crop
        resized = _resize_with_aspect(panel_img, w, h)
        
        if clip_polygon:
            # Masked paste for slanted/diagonal panels
            # clip_polygon is in global coords; convert to panel-local
            local_poly = [(px - x1, py - y1) for px, py in clip_polygon]
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).polygon(local_poly, fill=255)
            canvas.paste(resized, (x1, y1), mask)
        else:
            canvas.paste(resized, (x1, y1))
    else:
        # Stylised placeholder
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([x1, y1, x2, y2], fill=(218, 218, 218))
        draw.text((x1 + 8, y1 + 8), f"panel {x1},{y1}", fill=(160, 160, 160))

    # ── Inked Border ──────────────────────────────────────────────────────────
    if clip_polygon:
        _draw_inked_poly(canvas, clip_polygon, cfg)
    else:
        _draw_inked_rect(canvas, bbox, cfg)


def _draw_inked_rect(
    canvas: Image.Image,
    bbox: List[int],
    cfg: CompositorConfig,
) -> None:
    x1, y1, x2, y2 = bbox
    poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    _draw_inked_poly(canvas, poly, cfg)


def _draw_inked_poly(
    canvas: Image.Image,
    points: List[List[float]],
    cfg: CompositorConfig,
) -> None:
    """Draw a hand-inked style polygon border."""
    draw = ImageDraw.Draw(canvas)
    # Convert points to tuples and close the loop
    pts = [tuple(p) for p in points]
    inked_poly = _jittered_line_poly(pts + [pts[0]], wobble=1.2)
    
    # Base stroke
    draw.line(inked_poly, fill=cfg.border_color, width=cfg.border_width)
    
    # Secondary thinner stroke for "G-pen" character
    offset_poly = [(px + 0.8, py + 0.8) for px, py in inked_poly]
    draw.line(offset_poly, fill=cfg.border_color, width=1)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_layout(layout: Dict[str, Any]) -> None:
    if "panels" not in layout:
        raise ValueError("Layout missing required key: 'panels'")
    for idx, panel in enumerate(layout["panels"]):
        if "bbox" not in panel:
            raise ValueError(f"Panel {idx} is missing 'bbox'")
        bbox = panel["bbox"]
        if len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
            raise ValueError(f"Panel {idx} has invalid bbox: {bbox}")
        for bidx, bubble in enumerate(panel.get("bubbles", [])):
            if "bbox" in bubble and len(bubble["bbox"]) != 4:
                raise ValueError(
                    f"Bubble {bidx} in panel {idx} has invalid bbox"
                )


# ── Page metadata ─────────────────────────────────────────────────────────────

def _save_with_metadata(
    image: Image.Image,
    path: Path,
    metadata: Dict[str, Any],
) -> None:
    try:
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        for k, v in metadata.items():
            info.add_text(str(k), str(v))
        image.save(path, "PNG", pnginfo=info)
    except Exception as exc:
        logger.warning("Could not embed metadata: %s", exc)
        image.save(path, "PNG")


# ── Main compositor ────────────────────────────────────────────────────────────

def compose_page(
    layout: Dict[str, Any],
    panels_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cfg: CompositorConfig = CompositorConfig(),
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Image.Image:
    """
    Compose a single manga page from its layout JSON.

    Parameters
    ----------
    layout           : parsed layout JSON dict
    panels_dir       : directory containing per-panel PNG images
    output_dir       : where to save the full page and individual panels
    cfg              : CompositorConfig instance
    progress_callback: optional fn(current, total, message) for progress UI

    Returns
    -------
    PIL Image of the full composed page
    """
    if cfg.validate:
        _validate_layout(layout)

    page_id  = layout.get("page_id", "unknown")
    canvas_w = layout.get("canvas", {}).get("width",  cfg.canvas_w)
    canvas_h = layout.get("canvas", {}).get("height", cfg.canvas_h)
    panels   = layout.get("panels", [])

    logger.info("Composing page '%s'  (%d panels)", page_id, len(panels))

    canvas = Image.new("RGB", (canvas_w, canvas_h), cfg.canvas_bg)
    draw   = ImageDraw.Draw(canvas)

    total_steps = len(panels) + 1
    if progress_callback:
        progress_callback(0, total_steps, f"Starting {page_id}")

    for idx, panel in enumerate(panels):
        bbox    = panel.get("bbox")
        bubbles = panel.get("bubbles", [])

        if not bbox or len(bbox) != 4:
            logger.warning("Panel %d has missing/invalid bbox — skipped", idx)
            continue

        # ── Place panel artwork ───────────────────────────────────────────────
        panel_img    = _find_panel_image(panels_dir, page_id, idx)
        clip_poly    = panel.get("clip_polygon")
        is_bleed     = panel.get("bleed", False)

        if panel_img is None:
            logger.debug("No image for panel %d — using placeholder", idx)

        # Save CLEAN panel (without bubbles) before drawing text
        if cfg.save_panels and output_dir is not None:
            px1, py1, px2, py2 = bbox
            src_img = panel_img if panel_img is not None else None
            pw, ph  = px2 - px1, py2 - py1
            if pw > 0 and ph > 0:
                clean_panel = (
                    _resize_with_aspect(src_img, pw, ph)
                    if src_img is not None
                    else Image.new("RGB", (pw, ph), (218, 218, 218))
                )
                panel_out_dir = output_dir / "panels" / page_id
                panel_out_dir.mkdir(parents=True, exist_ok=True)
                clean_panel.save(panel_out_dir / f"panel_{idx}.png")

        _place_panel(canvas, panel_img, bbox, cfg, clip_polygon=clip_poly, bleed=is_bleed)

        # ── Draw speech bubbles ───────────────────────────────────────────────
        for bubble in bubbles:
            btype = bubble.get("type", "speech")
            text  = bubble.get("text", "").strip()
            bbbox = bubble.get("bbox")

            if not bbbox or len(bbbox) != 4 or not text:
                continue

            if btype == "narration":
                _draw_narration_box(draw, bbbox, text, cfg)
            else:
                _draw_ellipse_bubble(draw, bbbox, text, btype, bbox, cfg)

        if progress_callback:
            progress_callback(idx + 1, total_steps,
                              f"Panel {idx + 1}/{len(panels)}")

    # ── Save full page ────────────────────────────────────────────────────────
    if output_dir is not None:
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        out_path  = pages_dir / f"{page_id}_full.png"
        metadata  = {
            "page_id":     page_id,
            "panel_count": len(panels),
            "version":     "2.0",
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_with_metadata(canvas, out_path, metadata)
        logger.info("[SAVED] %s", out_path)

    if progress_callback:
        progress_callback(total_steps, total_steps, f"Done: {page_id}")

    return canvas


# ── Batch mode ────────────────────────────────────────────────────────────────

def _compose_one(args: Tuple) -> None:
    """Worker target for parallel batch processing."""
    lf, panels_dir, output_dir, cfg = args
    try:
        with open(lf, encoding="utf-8") as f:
            layout = json.load(f)
        compose_page(layout, panels_dir, output_dir, cfg)
    except Exception as exc:
        logger.error("Error processing %s: %s", lf.name, exc, exc_info=True)


def compose_batch(
    layout_dir: Path,
    panels_dir: Optional[Path],
    output_dir: Path,
    cfg: CompositorConfig = CompositorConfig(),
    max_workers: int = 4,
) -> None:
    layout_files = sorted(layout_dir.glob("*.json"))
    logger.info("Found %d layout files in %s", len(layout_files), layout_dir)

    work = [(lf, panels_dir, output_dir, cfg) for lf in layout_files]

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compose_one, w): w[0].name for w in work}
            done = 0
            for future in futures:
                future.result()   # re-raises exceptions from worker
                done += 1
                logger.info("[%d/%d] complete", done, len(layout_files))
    else:
        for i, w in enumerate(work, 1):
            logger.info("[%d/%d] %s", i, len(layout_files), w[0].name)
            _compose_one(w)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3e — compose manga pages from layout JSON + panel images"
    )
    parser.add_argument("--layout",      help="single layout JSON file")
    parser.add_argument("--layout_dir",  help="directory of layout JSON files (batch)")
    parser.add_argument("--panels",      help="directory containing panel PNG images")
    parser.add_argument("--output",      required=True, help="output directory")
    parser.add_argument("--workers",     type=int, default=1,
                        help="parallel worker threads for batch mode (default: 1)")
    parser.add_argument("--wobble",      type=int, default=6,
                        help="organic bubble jitter in px (default: 6)")
    parser.add_argument("--spikes",      type=int, default=20,
                        help="shout bubble spike count (default: 20)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip layout validation")
    parser.add_argument("--verbose",     action="store_true",
                        help="enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg            = CompositorConfig()
    cfg.organic_wobble   = args.wobble
    cfg.shout_spike_count = args.spikes
    cfg.validate         = not args.no_validate

    output_dir = Path(args.output)
    panels_dir = Path(args.panels) if args.panels else None
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.layout:
        with open(args.layout, encoding="utf-8") as f:
            layout = json.load(f)
        compose_page(layout, panels_dir, output_dir, cfg)

    elif args.layout_dir:
        compose_batch(
            Path(args.layout_dir),
            panels_dir,
            output_dir,
            cfg,
            max_workers=args.workers,
        )

    else:
        parser.error("Provide either --layout or --layout_dir")


if __name__ == "__main__":
    main()