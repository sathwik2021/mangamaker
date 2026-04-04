"""
run_e2e_test_enhanced.py -- Enhanced End-to-End Pipeline
Text -> Beats -> Layout -> SD LoRA Panel Images -> Screentone -> Compositor -> Final Page
"""

import gc
import os
import sys
import json
import time
import logging
import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import threading
import signal

import numpy as np
from PIL import Image, ImageFilter
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import cv2
from scipy import ndimage

# ── Optional: CLIP for semantic scoring ────────────────────────────────────
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP not available — semantic scoring disabled. "
                  "Install with: pip install transformers")

# ── Load .env file if present ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("manga_pipeline_enhanced")

# ── Feature flags ──────────────────────────────────────────────────────────────
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
USE_CLIP = os.getenv("USE_CLIP", "1") == "1" and CLIP_AVAILABLE
USE_CONTROLNET = os.getenv("USE_CONTROLNET", "0") == "1"

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
STEP2_DIR      = PROJECT_ROOT / "step-2-layout"
STEP3_DIR      = PROJECT_ROOT / "step-3"
TEST_OUT       = PROJECT_ROOT / "test_output"
PANELS_OUT     = TEST_OUT / "panels"
LAYOUT_OUT     = TEST_OUT / "layout"
PAGES_OUT      = TEST_OUT / "pages"
CHECKPOINT_DIR = TEST_OUT / "checkpoints"
PROMPTS_OUT    = TEST_OUT / "prompts"
CACHE_DIR      = TEST_OUT / "cache"
METRICS_OUT    = TEST_OUT / "metrics"
STEP1_DIR      = PROJECT_ROOT / "Step-1" / "code"

sys.path.insert(0, str(STEP1_DIR))
sys.path.insert(0, str(STEP2_DIR))
sys.path.insert(0, str(STEP3_DIR))

# ── Import pipeline modules ────────────────────────────────────────────────
try:
    from model_client import generate, get_current_model
    from layout_generator import convert_beats_to_layout
    from compositor import compose_page, CompositorConfig
    from text_extractor import TextExtractor
    from dialog_mapper import DialogMapper
except ImportError as e:
    logger.error(f"Failed to import a pipeline module: {e}")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    canvas_width:  int   = 1800
    canvas_height: int   = 2400
    panel_gap:     int   = 10
    min_panels:    int   = 4
    max_panels:    int   = 7
    sd_steps:           int   = 40
    sd_guidance_scale:  float = 9.0
    sd_max_dim:         int   = 768
    sd_seed_base:       int   = 42
    num_candidates:     int   = 2
    max_retries:        int   = 3
    use_lora:           bool  = True
    use_context_prompt: bool  = True
    use_continuity:     bool  = True
    use_clip_scoring:   bool  = USE_CLIP
    use_controlnet:     bool  = USE_CONTROLNET
    screentone_enabled:    bool  = True
    screentone_dot_radius: int   = 2
    screentone_spacing:    int   = 6
    screentone_threshold:  int   = 128
    screentone_strength:   float = 0.55
    screentone_adaptive:   bool  = True
    layout_min_coverage: float = 0.90
    layout_max_coverage: float = 1.02
    image_min_contrast:  float = 25.0
    image_min_mean:      float = 20.0
    image_max_mean:      float = 235.0
    image_aspect_tol:    float = 0.15
    quality_min_score:   float = 60.0
    global_style: str = "clean manga lineart, G-pen linework, professional screentone"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            sd_steps           = int(os.getenv("SD_STEPS",          cls.sd_steps)),
            sd_guidance_scale  = float(os.getenv("SD_GUIDANCE",     cls.sd_guidance_scale)),
            sd_seed_base       = int(os.getenv("SD_SEED",           cls.sd_seed_base)),
            num_candidates     = int(os.getenv("NUM_CANDIDATES",    cls.num_candidates)),
            max_retries        = int(os.getenv("MAX_RETRIES",       cls.max_retries)),
        )

CONFIG = PipelineConfig.from_env()

@dataclass
class PipelineMetrics:
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    step_times: Dict[str, float] = field(default_factory=dict)
    panel_quality_scores: List[float] = field(default_factory=list)
    panel_clip_scores: List[float] = field(default_factory=list)
    generation_attempts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    def record_step(self, step_name: str, duration: float) -> None:
        self.step_times[step_name] = duration
    def record_panel_quality(self, score: float, clip_score: Optional[float] = None) -> None:
        self.panel_quality_scores.append(score)
        if clip_score is not None: self.panel_clip_scores.append(clip_score)
    def report(self) -> Dict[str, Any]:
        return {"total_seconds": ((self.end_time or datetime.now()) - self.start_time).total_seconds()}

@dataclass
class CharacterEmbedding:
    name: str; description: str; appearance_tokens: str; seen_count: int = 0
    consistency_score: float = 1.0; last_panel_idx: int = -1

@contextmanager
def log_stage(name: str):
    start = time.time(); logger.info(f"▶ Starting: {name}")
    try: yield
    finally: logger.info(f"✅ Done: {name} ({time.time() - start:.1f}s)")

def save_checkpoint(stage: str, data: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{stage}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "timestamp": time.time(), "data": data}, f, indent=2)

def load_checkpoint(stage: str) -> Optional[dict]:
    path = CHECKPOINT_DIR / f"{stage}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload["data"]
    return None

def image_sharpness_score(img: Image.Image) -> float:
    try:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        return float(cv2.Laplacian(arr, cv2.CV_64F).var())
    except Exception:
        arr = np.array(img.convert("L"), dtype=np.float32)
        return float(np.std(arr) * 10)

def extract_character_tokens(description: str, n: int = 3) -> str:
    parts = [p.strip() for p in description.split(",") if p.strip()]
    return ", ".join(parts[:n]) if parts else description[:60]

def panels_overlap(b1: List[int], b2: List[int]) -> bool:
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

class CLIPScorer:
    def __init__(self):
        if not CLIP_AVAILABLE: raise RuntimeError("CLIP not available")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
    def score(self, image: Image.Image, text: str) -> float:
        truncated_text = text[:100]
        with torch.no_grad():
            inputs = self.processor(text=[truncated_text], images=image, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return torch.sigmoid(outputs.logits_per_image[0, 0]).item()

class CharacterMemory:
    def __init__(self):
        self.characters: Dict[str, CharacterEmbedding] = {}
    def register(self, name: str, description: str, appearance: str) -> None:
        if name in self.characters:
            self.characters[name].appearance_tokens = appearance
            self.characters[name].seen_count += 1
        else:
            self.characters[name] = CharacterEmbedding(name=name, description=description, appearance_tokens=appearance)

def _build_screentone_mask(width: int, height: int, dot_radius: int, spacing: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    ys, xs = np.arange(0, height, spacing), np.arange(0, width, spacing)
    r2 = dot_radius ** 2
    for cy in ys:
        for cx in xs:
            y0, y1 = max(0, cy - dot_radius), min(height, cy + dot_radius + 1)
            x0, x1 = max(0, cx - dot_radius), min(width, cx + dot_radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r2
    return mask

def apply_screentone(img: Image.Image, dot_radius: int = 2, spacing: int = 6, threshold: int = 128, strength: float = 0.55, adaptive: bool = True) -> Image.Image:
    img_rgb = img.convert("RGB"); arr = np.array(img_rgb, dtype=np.float32); gray = np.array(img_rgb.convert("L"), dtype=np.float32)
    w, h = img_rgb.size; dot_mask = _build_screentone_mask(w, h, dot_radius, spacing); shadow_mask = gray < threshold
    apply_mask = (dot_mask & shadow_mask).astype(np.float32); apply_mask_3 = apply_mask[:, :, np.newaxis]
    blended = np.clip(arr - (apply_mask_3 * strength * 60.0), 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")

class QualityAssessor:
    def assess_image(self, img_path: Path, bbox: List[int]) -> Dict[str, Any]:
        img = Image.open(img_path); arr = np.array(img.convert("L"), dtype=np.float32)
        mean, std = float(arr.mean()), float(arr.std())
        laplacian = cv2.Laplacian(arr, cv2.CV_64F).var()
        score = 100.0
        if std < CONFIG.image_min_contrast: score -= 30
        if mean < CONFIG.image_min_mean or mean > CONFIG.image_max_mean: score -= 20
        return {"path": str(img_path), "score": score, "passed": score >= CONFIG.quality_min_score}

MOOD_STYLE_MAP = {
    "tense": "dynamic dutch-angle perspective, dramatic chiaroscuro lighting, heavy G-pen ink shadows",
    "action": "extreme dynamic angle, kinetic speed lines, motion blur, bold G-pen linework",
    "calm": "soft even lighting, peaceful atmosphere, detailed hatched background, gentle Maru-pen fine linework",
    "emotional": "Rembrandt 3/4 lighting, expressive Maru-pen close-up, sparkle and shine effects on eyes",
    "neutral": "balanced professional manga composition, consistent G-pen linework, clean precise inking",
}

def _smart_truncate(prompt: str, max_tokens: int = 75) -> str:
    clauses = [c.strip() for c in prompt.split(",") if c.strip()]
    res, count = [], 0
    for c in clauses:
        w = c.split()
        if count + len(w) > max_tokens: break
        res.append(c); count += len(w)
    return ", ".join(res) if res else " ".join(prompt.split()[:max_tokens])

def _aspect_ratio_dims(bbox: List[int], max_dim: int) -> Tuple[int, int]:
    pw, ph = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]); r = pw / ph
    if r > 1.35: w, h = 768, 512
    elif r < 0.74: w, h = 512, 768
    else: w, h = 640, 640
    return max(64, min(max_dim, (w // 8) * 8)), max(64, min(max_dim, (h // 8) * 8))

def build_prompt(panel: dict, **kwargs) -> str:
    tags = panel.get("description", "manga scene"); mood = panel.get("mood", "neutral")
    style = MOOD_STYLE_MAP.get(mood, MOOD_STYLE_MAP["neutral"])
    prompt = f"manga style, monochrome, {tags}, {style}, masterpiece, best quality"
    return _smart_truncate(prompt, max_tokens=75)

def build_negative_prompt(panel: dict) -> str:
    return "color, blurry, low quality, bad anatomy, text, watermark, multiple people"

def _save_prompt(idx: int, page_id: str, prompt: str, neg_prompt: str) -> None:
    PROMPTS_OUT.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_OUT / f"{page_id}_panel_{idx}_prompt.txt", "w", encoding="utf-8") as f:
        f.write(f"POSITIVE:\n{prompt}\n\nNEGATIVE:\n{neg_prompt}\n")

def step1_extract_beats(text: str, chunk_id: str = "chunk_001", max_retries: int = 3) -> dict:
    prompt = f"""Extract manga narrative beats from the following text.
Return ONLY raw JSON in this exact format, with no markdown code blocks:
{{
  "beats": [
    {{
      "id": "beat_1",
      "description": "Visual description of what happens",
      "mood": "neutral"
    }}
  ]
}}

Text: {text}"""
    for attempt in range(max_retries):
        try:
            res = generate(prompt); data = res if isinstance(res, dict) else json.loads(res.strip("`").strip("json").strip())
            return data
        except Exception as e:
            if attempt == max_retries - 1: raise e
            time.sleep(2 ** attempt)

def load_sd_pipeline() -> StableDiffusionPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"; dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None)
    if device == "cuda": pipe.to(device)
    else: pipe.enable_attention_slicing()
    return pipe

def step3_generate_panels(layout: dict, pipe: StableDiffusionPipeline, metrics: PipelineMetrics) -> Path:
    page_id = layout.get("page_id", "page_1"); out_dir = PANELS_OUT / page_id; out_dir.mkdir(parents=True, exist_ok=True)
    for idx, panel in enumerate(layout.get("panels", [])):
        prompt = build_prompt(panel); neg_prompt = build_negative_prompt(panel)
        bbox = panel.get("bbox", [0, 0, 512, 512]); w, h = _aspect_ratio_dims(bbox, CONFIG.sd_max_dim)
        res = pipe(prompt, negative_prompt=neg_prompt, width=w, height=h, num_inference_steps=CONFIG.sd_steps)
        img = res.images[0].resize((bbox[2]-bbox[0], bbox[3]-bbox[1]), Image.LANCZOS)
        if CONFIG.screentone_enabled: img = apply_screentone(img)
        img.save(out_dir / f"panel_{idx}.png")
    return out_dir

def step3_composite(layout: dict, panels_dir: Path) -> Path:
    PAGES_OUT.mkdir(parents=True, exist_ok=True); cfg = CompositorConfig(save_panels=False)
    compose_page(layout=layout, panels_dir=panels_dir, output_dir=TEST_OUT, cfg=cfg)
    return PAGES_OUT / f"{layout.get('page_id', 'page_1')}_full.png"

def validate_layout(layout: dict) -> bool:
    return True

def main() -> None:
    print("\n" + "=" * 70); print("  🎨 ENHANCED MANGA PIPELINE (CLI)"); print("=" * 70)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key: logger.error("GEMINI_API_KEY not set!"); sys.exit(1)
    test_file = PROJECT_ROOT / "test_input.txt"
    with open(test_file, "r", encoding="utf-8") as f: raw_text = f.read().strip()
    text = " ".join(raw_text.split()[:800])
    run_start = time.time(); metrics = PipelineMetrics()
    beats = step1_extract_beats(text); layout = convert_beats_to_layout(beats)
    pipe = load_sd_pipeline() if not DRY_RUN else None
    panels_dir = step3_generate_panels(layout, pipe, metrics)
    final_page = step3_composite(layout, panels_dir)
    print(f"\n  ✅ COMPLETE! Final Page → {final_page}")

if __name__ == "__main__":
    main()