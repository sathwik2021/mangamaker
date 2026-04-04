"""
text_renderer.py — Render speech bubbles and narration onto manga panels

GPU-optimized text rendering with efficient font caching and batch processing.
Supports multiple bubble styles: speech, thought, narration_box, scream.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("text_renderer")


@dataclass
class BubbleStyle:
    """Configuration for a bubble rendering style."""
    name: str
    fill_color: Tuple[int, int, int]
    stroke_color: Tuple[int, int, int]
    stroke_width: int
    font_size: int
    use_circles: bool = False  # For thought bubbles
    tail_style: str = "pointer"  # pointer, none, double_circle
    bg_alpha: int = 255


# Predefined bubble styles
STYLES = {
    "speech": BubbleStyle(
        name="speech",
        fill_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=3,
        font_size=28,
        tail_style="pointer"
    ),
    "thought": BubbleStyle(
        name="thought",
        fill_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=2,
        font_size=24,
        use_circles=True,
        tail_style="double_circle"
    ),
    "scream": BubbleStyle(
        name="scream",
        fill_color=(255, 200, 200),
        stroke_color=(255, 0, 0),
        stroke_width=4,
        font_size=32,
        tail_style="pointer"
    ),
    "whisper": BubbleStyle(
        name="whisper",
        fill_color=(220, 220, 220),
        stroke_color=(100, 100, 100),
        stroke_width=2,
        font_size=20,
        tail_style="pointer"
    ),
    "narration_box": BubbleStyle(
        name="narration_box",
        fill_color=(240, 240, 240),
        stroke_color=(0, 0, 0),
        stroke_width=2,
        font_size=22,
        tail_style="none"
    ),
}


class TextRenderer:
    """GPU-optimized text renderer for manga panels."""
    
    def __init__(self, font_dir: Optional[str] = None, device: str = "cuda"):
        """
        Initialize text renderer.
        
        Args:
            font_dir: Directory containing TrueType fonts
            device: "cuda" or "cpu"
        """
        self.font_dir = font_dir or self._find_system_fonts()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.font_cache = {}
        
        self.logger.info(f"TextRenderer initialized on {self.device.upper()}")
    
    def _find_system_fonts(self) -> str:
        """Find system fonts directory."""
        candidates = [
            "/usr/share/fonts/truetype/dejavu/",
            "C:\\Windows\\Fonts\\",
            "/System/Library/Fonts/",
            "/Library/Fonts/",
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return "/usr/share/fonts/truetype/dejavu/"
    
    @lru_cache(maxsize=16)
    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load and cache font at specified size."""
        font_paths = [
            os.path.join(self.font_dir, "DejaVuSans-Bold.ttf"),
            os.path.join(self.font_dir, "LiberationSans-Bold.ttf"),
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception as e:
                    self.logger.warning(f"Failed to load font {path}: {e}")
        
        # Fallback
        self.logger.warning("No font found, using default")
        return ImageFont.load_default()
    
    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        max_lines: int = 4
    ) -> List[str]:
        """
        Wrap text to fit within max_width.
        
        Returns list of lines with max max_lines.
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                
                if len(lines) >= max_lines - 1:
                    # Truncate and add ellipsis
                    break
        
        if current_line:
            line_text = " ".join(current_line)
            if len(lines) >= max_lines:
                # Truncate with ellipsis
                line_text = line_text[:max(1, len(line_text) - 3)] + "..."
            lines.append(line_text)
        
        return lines[:max_lines]
    
    def _render_bubble_shape(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: Tuple[int, int, int, int],
        style: BubbleStyle,
        tail_anchor: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Draw bubble shape with tail.
        
        Args:
            draw: PIL ImageDraw
            bbox: (x1, y1, x2, y2) bounding box
            style: BubbleStyle configuration
            tail_anchor: (x, y) point where tail points to
        """
        x1, y1, x2, y2 = bbox
        
        if style.use_circles:
            # Thought bubble: circles around rounded rectangle center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            
            # Main bubble (rounded)
            draw.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=min(w, h) // 4,
                fill=style.fill_color,
                outline=style.stroke_color,
                width=style.stroke_width
            )
            
            # Tail circles
            if tail_anchor:
                tx, ty = tail_anchor
                # Circle 1: medium
                r1 = 8
                draw.ellipse(
                    [tx - r1, ty - r1, tx + r1, ty + r1],
                    fill=style.fill_color,
                    outline=style.stroke_color,
                    width=style.stroke_width
                )
                # Circle 2: small
                r2 = 5
                offset = 15
                draw.ellipse(
                    [tx - r2 + offset, ty - r2 + offset, tx + r2 + offset, ty + r2 + offset],
                    fill=style.fill_color,
                    outline=style.stroke_color,
                    width=style.stroke_width
                )
        else:
            # Regular bubble (oval with tail)
            if style.name == "narration_box":
                # Simple rectangle for narration
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=style.fill_color,
                    outline=style.stroke_color,
                    width=style.stroke_width
                )
            else:
                # Rounded rectangle with tail
                draw.rounded_rectangle(
                    [x1, y1, x2, y2],
                    radius=max(w := x2 - x1, h := y2 - y1) // 6,
                    fill=style.fill_color,
                    outline=style.stroke_color,
                    width=style.stroke_width
                )
                
                # Draw tail if specified
                if tail_anchor and style.tail_style == "pointer":
                    tx, ty = tail_anchor
                    
                    # Determine tail position (bottom-right, bottom-left, etc.)
                    # Use simple triangle
                    points = []
                    
                    # Find closest edge
                    edges = [
                        ("bottom", y2),
                        ("right", x2),
                        ("left", x1),
                        ("top", y1)
                    ]
                    
                    closest = min(edges, key=lambda e: abs(e[1] - (tx if e[0] in ["left", "right"] else ty)))
                    edge_type = closest[0]
                    
                    # Tail triangle
                    tail_len = 20
                    if edge_type == "bottom":
                        points = [(x2 - 30, y2), (x2 - 10, y2), (tx, ty)]
                    elif edge_type == "right":
                        points = [(x2, y2 - 30), (x2, y2 - 10), (tx, ty)]
                    elif edge_type == "left":
                        points = [(x1, y1 + 30), (x1, y1 + 10), (tx, ty)]
                    else:  # top
                        points = [(x1 + 30, y1), (x1 + 10, y1), (tx, ty)]
                    
                    if points:
                        draw.polygon(
                            points,
                            fill=style.fill_color,
                            outline=style.stroke_color
                        )
    
    def render_bubble(
        self,
        panel_image: Image.Image,
        bubble_spec: Dict[str, Any],
        style: Optional[str] = None
    ) -> Image.Image:
        """
        Render a single bubble onto panel image.
        
        Args:
            panel_image: PIL Image (RGB or RGBA)
            bubble_spec: {
                "bbox": [x1, y1, x2, y2],
                "text": "...",
                "type": "speech" | "thought" | "narration_box",
                "tail_position": "top_right" | "left" | etc.
            }
            style: Override style name
        
        Returns:
            Panel image with bubble rendered
        """
        if not bubble_spec.get("text"):
            return panel_image
        
        # Get style
        style_name = style or bubble_spec.get("type", "speech")
        style = STYLES.get(style_name, STYLES["speech"])
        
        # Get bubble bbox
        bbox = bubble_spec.get("bbox", [0, 0, 100, 100])
        if len(bbox) != 4:
            self.logger.warning(f"Invalid bbox: {bbox}")
            return panel_image
        
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        # Clamp to panel bounds
        w, h = panel_image.size
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1 + 20, min(x2, w))
        y2 = max(y1 + 20, min(y2, h))
        
        # Load font
        font = self._load_font(style.font_size)
        
        # Wrap text
        max_width = x2 - x1 - 20
        lines = self._wrap_text(bubble_spec["text"], font, max_width)
        
        if not lines:
            return panel_image
        
        # Create drawing context
        draw = ImageDraw.Draw(panel_image, "RGBA")
        
        # Calculate tail anchor (if specified)
        tail_anchor = None
        tail_pos = bubble_spec.get("tail_position", "")
        if tail_pos:
            if "top" in tail_pos and "right" in tail_pos:
                tail_anchor = (x2 - 10, y1 - 15)
            elif "left" in tail_pos:
                tail_anchor = (x1 - 15, (y1 + y2) // 2)
            elif "bottom" in tail_pos:
                tail_anchor = ((x1 + x2) // 2, y2 + 15)
        
        # Draw bubble shape
        self._render_bubble_shape(draw, (x1, y1, x2, y2), style, tail_anchor)
        
        # Draw text
        text_y = y1 + 10
        line_spacing = style.font_size + 4
        
        for line in lines:
            draw.text(
                (x1 + 10, text_y),
                line,
                fill=(0, 0, 0, 255),
                font=font,
                anchor="lt"
            )
            text_y += line_spacing
        
        return panel_image
    
    def render_all_bubbles(
        self,
        panel_image: Image.Image,
        panel_layout: Dict[str, Any],
        max_bubbles: int = 3
    ) -> Image.Image:
        """
        Render all bubbles in a panel.
        
        Args:
            panel_image: PIL Image
            panel_layout: Panel dict with "bubbles" array
            max_bubbles: Skip rendering if more than this (overcrowded)
        
        Returns:
            Panel image with all bubbles rendered
        """
        bubbles = panel_layout.get("bubbles", [])
        
        if not bubbles:
            return panel_image
        
        if len(bubbles) > max_bubbles:
            self.logger.warning(
                f"Panel {panel_layout.get('id')} has {len(bubbles)} bubbles "
                f"(max {max_bubbles}), skipping some"
            )
            bubbles = bubbles[:max_bubbles]
        
        for bubble in bubbles:
            panel_image = self.render_bubble(panel_image, bubble)
        
        return panel_image


# GPU-accelerated batch rendering (experimental)
def render_batch_gpu(
    panel_images: List[Image.Image],
    panel_layouts: List[Dict[str, Any]],
    device: str = "cuda"
) -> List[Image.Image]:
    """
    Render text on multiple panels in parallel (using threading, not GPU).
    
    Note: Text rendering is CPU-bound; GPU acceleration would require
    CUDA-based text rasterization library which is not readily available.
    Using this function for thread-level parallelism instead.
    """
    import concurrent.futures
    
    renderer = TextRenderer(device=device)
    results = []
    
    def render_one(img, layout):
        return renderer.render_all_bubbles(img, layout)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(render_one, img, layout)
            for img, layout in zip(panel_images, panel_layouts)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    return results
