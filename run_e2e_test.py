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

import re
import numpy as np
from PIL import Image, ImageFilter
import torch
from diffusers import StableDiffusionPipeline
# PeftModel import removed — LoRA is loaded via pipe.load_lora_weights(), not PeftModel
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

# ── LoRA configuration (FIX 1) ─────────────────────────────────────────────────
# Point MANGA_LORA_PATH at the output dir from step-3/train_lora.py.
# If the path does not exist, load_sd_pipeline() hard-fails — never silently
# falls back to stock SD1.5 (that is how the LoRA absence went unnoticed).
# Default now uses 1.0 because it produces noticeably stronger manga-style
# character renderings in the experimental comparisons.
LORA_PATH  = os.getenv("MANGA_LORA_PATH", "./step-3/results/lora_output/final_lora")
LORA_SCALE = float(os.getenv("MANGA_LORA_SCALE", "1.0"))

sys.path.insert(0, str(STEP1_DIR))
sys.path.insert(0, str(STEP2_DIR))
sys.path.insert(0, str(STEP3_DIR))

# ── Import pipeline modules ────────────────────────────────────────────────
try:
    from model_client import generate, get_current_model
    from layout_generator import convert_beats_to_layout, clean_json_response  # FIX 5
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


def configure_pipeline(options: Dict[str, Any]) -> None:
    """Override runtime pipeline configuration from request options."""
    if not isinstance(options, dict):
        return

    if "guidance_scale" in options:
        try:
            CONFIG.sd_guidance_scale = float(options["guidance_scale"])
            logger.info("Configured SD guidance_scale from options: %s", CONFIG.sd_guidance_scale)
        except Exception as exc:
            logger.warning(
                "Could not parse guidance_scale option '%s': %s",
                options["guidance_scale"], exc,
            )

    if "sd_guidance_scale" in options:
        try:
            CONFIG.sd_guidance_scale = float(options["sd_guidance_scale"])
            logger.info("Configured SD guidance_scale from options: %s", CONFIG.sd_guidance_scale)
        except Exception as exc:
            logger.warning(
                "Could not parse sd_guidance_scale option '%s': %s",
                options["sd_guidance_scale"], exc,
            )

    # Preserve actual applied guidance scale for metadata
    options["applied_guidance_scale"] = str(CONFIG.sd_guidance_scale)

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

def _smart_truncate(prompt: str, tokenizer=None, max_tokens: int = 75) -> str:
    """
    Truncate prompt to fit under CLIP's 77-token limit (75 content tokens +
    2 special tokens), using the actual tokenizer when available.
    """
    if tokenizer is None:
        logger.warning(
            "_smart_truncate called without tokenizer — using word-count fallback, may overflow CLIP limit"
        )
        words = prompt.split()
        return " ".join(words[:max_tokens])

    tokenized = tokenizer(prompt, truncation=False)
    token_ids = tokenized.get("input_ids", [])
    if len(token_ids) <= max_tokens + 2:
        return prompt

    truncated_ids = token_ids[: max_tokens + 1]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text.strip().rstrip(",")


def _simplify_abstract_description(tags_body: str) -> str:
    """Simplify and anchor abstract no-human prompts for better SD consistency."""
    abstract_terms = [
        "dreamworld",
        "dream world",
        "dreamlike",
        "surreal",
        "otherworldly",
        "ethereal",
        "abstract",
        "mystical",
        "psychedelic",
        "gaudy",
        "dazzling",
        "murmurs",
        "shattered",
        "throbbing",
        "painful",
        "agony",
        "disorientation",
        "blurred",
        "hazy",
        "phantasmagoric",
        "floating",
        "twisted",
        "fractured",
        "shimmering",
        "melting",
        "liquid",
    ]
    for term in abstract_terms:
        tags_body = re.sub(rf"\b{re.escape(term)}\b", "", tags_body, flags=re.IGNORECASE)

    tags_body = re.sub(r"\s*,\s*", ", ", tags_body.strip())
    tags_body = re.sub(r",{2,}", ",", tags_body)
    tags_body = tags_body.strip(" ,")

    if re.search(r"\bno humans\b|\bno human\b", tags_body, flags=re.IGNORECASE):
        if not re.search(r"\b(background|landscape|sky|clouds|moon|stars|horizon|forest|city|street|mountain|river|water|stone|bridge|door|temple|altar|hall|room|mist|fog)\b", tags_body, flags=re.IGNORECASE):
            if tags_body:
                tags_body = f"{tags_body}, background, simple horizon, subtle foreground object"
            else:
                tags_body = "no humans, background, simple horizon, subtle foreground object"

    return tags_body


def _aspect_ratio_dims(bbox: List[int], max_dim: int) -> Tuple[int, int]:
    pw, ph = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]); r = pw / ph
    if r > 1.35: w, h = 768, 512
    elif r < 0.74: w, h = 512, 768
    else: w, h = 640, 640
    return max(64, min(max_dim, (w // 8) * 8)), max(64, min(max_dim, (h // 8) * 8))

def build_prompt(panel: dict, **kwargs) -> str:
    tags = panel.get("description", "manga scene")
    mood = panel.get("mood", "neutral")
    style = MOOD_STYLE_MAP.get(mood, MOOD_STYLE_MAP["neutral"])

    anchor = "manga style, monochrome, screentone, masterpiece, best quality"

    tags_body = tags
    for dup in ("manga style, monochrome,", "manga style, monochrome"):
        if tags_body.startswith(dup):
            tags_body = tags_body[len(dup):].lstrip(", ")
            break

    if re.search(r"\b(no humans|no human|dreamworld|dreamlike|surreal|abstract)\b", tags_body, flags=re.IGNORECASE):
        tags_body = _simplify_abstract_description(tags_body)

    if tags_body:
        full_prompt = f"{anchor}, {tags_body}, {style}"
    else:
        full_prompt = f"{anchor}, {style}"

    return _smart_truncate(full_prompt, tokenizer=kwargs.get("tokenizer"), max_tokens=75)

def build_negative_prompt(panel: dict) -> str:
    return "color, blurry, low quality, bad anatomy, text, watermark, multiple people"

def _save_prompt(idx: int, page_id: str, prompt: str, neg_prompt: str) -> None:
    PROMPTS_OUT.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_OUT / f"{page_id}_panel_{idx}_prompt.txt", "w", encoding="utf-8") as f:
        f.write(f"POSITIVE:\n{prompt}\n\nNEGATIVE:\n{neg_prompt}\n")

def step1_extract_beats(text: str, chunk_id: str = "chunk_001", max_retries: int = 3) -> dict:
    # FIX 10: Strengthened grounding constraints to prevent pretraining-knowledge
    # contamination. The LLM may have prior knowledge of popular novels (e.g. Lord
    # of Mysteries) and blend memorized later-book content into extractions.
    prompt = f"""Extract manga narrative beats from the following text.

CRITICAL GROUNDING RULES — you MUST follow ALL of these:
1. Use ONLY information explicitly present in the provided text below.
2. Do NOT add any dialogue, thoughts, words, names, objects, or events that are not
   directly stated or clearly implied in the provided text.
3. If the source text does not contain dialogue or internal thoughts/monologue for a beat,
   do NOT invent any. Use an empty string for dialogue in that case.
4. Do NOT draw on any prior knowledge you may have about these characters,
   this story, or this author's other works. Treat the text as if you have
   never encountered it before.
5. Every "description" field must be traceable to a specific sentence or
   phrase in the provided text.

Return ONLY raw JSON in this exact format, with no markdown code blocks:
{{
  "beats": [
    {{
      "id": "beat_1",
      "description": "Visual description of what happens",
      "dialogue": "Exact quote of spoken dialogue or internal monologue/thoughts from the text, or empty string if none",
      "mood": "neutral"
    }}
  ]
}}

Text: {text}"""
    for attempt in range(max_retries):
        try:
            res = generate(prompt)
            # FIX 5: clean_json_response() correctly strips ```json...``` fences by
            # splitting on newlines. The old .strip("json") stripped individual
            # characters j/s/o/n from string ends, corrupting valid responses.
            data = res if isinstance(res, dict) else json.loads(clean_json_response(res))
            return data
        except Exception as e:
            if attempt == max_retries - 1: raise e
            time.sleep(2 ** attempt)

def load_sd_pipeline() -> StableDiffusionPipeline:
    # FIX 1: Hard-fail if LoRA weights are absent. Never silently fall back to
    # stock SD1.5 — that is how the missing LoRA went unnoticed in production.
    # FIX 0: Model ID corrected to community mirror (runwayml org deleted Aug 2024).
    if not os.path.isdir(LORA_PATH):
        raise FileNotFoundError(
            f"LoRA weights not found at '{LORA_PATH}'. "
            f"Set MANGA_LORA_PATH env var to the correct path, "
            f"or train the LoRA first (see step-3/train_lora.py). "
            f"Refusing to silently fall back to stock SD1.5."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"; dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None)
    if device == "cuda":
        pipe.to(device)
    else:
        pipe.enable_attention_slicing()
    pipe.load_lora_weights(LORA_PATH)
    logger.info(
        "Loaded LoRA from '%s' at scale %s, pipe_id=%s, device=%s",
        LORA_PATH,
        LORA_SCALE,
        id(pipe),
        device,
    )
    return pipe

def step3_generate_panels(layout: dict, pipe: StableDiffusionPipeline, metrics: PipelineMetrics) -> Path:
    page_id = layout.get("page_id", "page_1"); out_dir = PANELS_OUT / page_id; out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "SD GENERATE start page=%s panels=%d pipe_id=%s",
        page_id,
        len(layout.get("panels", [])),
        id(pipe),
    )
    for idx, panel in enumerate(layout.get("panels", [])):
        prompt = build_prompt(panel, tokenizer=pipe.tokenizer)
        neg_prompt = build_negative_prompt(panel)
        bbox = panel.get("bbox", [0, 0, 512, 512]); w, h = _aspect_ratio_dims(bbox, CONFIG.sd_max_dim)
        device = getattr(pipe, 'device', None)
        generator = None
        try:
            if device is not None:
                seed = CONFIG.sd_seed_base + idx
                generator = torch.Generator(device=device).manual_seed(seed)
        except Exception:
            generator = None

        # FIX 1: pass LORA_SCALE via cross_attention_kwargs so the adapter
        # is applied at the configured strength, not silently ignored.
        # FIX 11: Actually pass guidance_scale from config. Previously this
        # kwarg was missing, silently falling back to diffusers default (7.5).
        # CONFIG.sd_guidance_scale (default 9.0, env-overridable via SD_GUIDANCE)
        # is now respected. For SD1.5 compound prompts, 7-9 is the sweet spot;
        # values ≥15 cause CFG over-drive / compositional binding failures.
        logger.info(
            "SD GENERATE page=%s panel=%s idx=%d prompt_len=%d width=%d height=%d "
            "cross_attention_kwargs={'scale': %s} guidance_scale=%s seed=%s pipe_id=%s",
            page_id,
            panel.get("id"),
            idx,
            len(prompt.split()),
            w,
            h,
            LORA_SCALE,
            CONFIG.sd_guidance_scale,
            seed if generator is not None else 'none',
            id(pipe),
        )
        res = pipe(
            prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            num_inference_steps=CONFIG.sd_steps,
            guidance_scale=CONFIG.sd_guidance_scale,
            cross_attention_kwargs={"scale": LORA_SCALE},
            generator=generator,
        )
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