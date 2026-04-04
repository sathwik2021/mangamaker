# 🚀 Enhanced Manga Pipeline — Quick Start Guide

## Installation

### 1. Clone & Setup
```bash
# Create virtual environment
python -m venv venv_enhanced
source venv_enhanced/bin/activate  # Windows: venv_enhanced\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers peft accelerate pillow numpy opencv-python scipy
pip install transformers  # For CLIP (optional but recommended)
pip install python-dotenv click tqdm psutil

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('CLIP available!')"
```

### 2. API Setup
```bash
# Create .env file in project root
cat > .env << 'EOF'
GEMINI_API_KEY=AIza...your_key_here...
EOF

# Or set environment variable
export GEMINI_API_KEY="AIza...your_key_here..."
```

---

## Usage

### Basic Run (Recommended Default)
```bash
python run_e2e_test_enhanced.py
```

Output:
```
======================================================================
  🎨 ENHANCED MANGA PIPELINE
======================================================================
  Config: 2 candidates, 3 retries, CLIP=True, LoRA=True

▶ Starting: Beat Extraction
✅ Done: Beat Extraction (8.2s)

▶ Starting: Layout Generation
✅ Done: Layout Generation (3.1s)

▶ Starting: Panel Generation
  Panel 1/5: Klein discovering the secret... 
    Prompt (72 tokens): manga panel, medium shot of Klein (brown...
    Attempt 1/3...
    ✅ Selected candidate 1 (score=0.85)
    📁 Saved: panel_0.png (640×960px)
    ✅ Quality score: 85.2

  Panel 2/5: Action scene with Azik...
    ...

▶ Starting: Compositing
✅ Done: Compositing (11.0s)

======================================================================
  ✅ PIPELINE COMPLETE!
======================================================================

  Beats         → test_output/beats.json
  Layout        → test_output/layout/layout.json
  Panels        → test_output/panels/page_chunk_001/
  Final Page    → test_output/pages/page_chunk_001_full.png
  Prompts       → test_output/prompts/
  Metrics       → test_output/metrics/metrics.json

  ⏱️  Total Time    → 432.5s
  Quality Score → 82.3/100
  CLIP Score    → 0.743
```

---

## Configuration Presets

### 1. Quick Test (2 min)
```bash
DRY_RUN=1 \
SD_STEPS=10 \
NUM_CANDIDATES=1 \
python run_e2e_test_enhanced.py
```
- ✅ Skips SD generation
- ✅ Tests beat extraction, layout, compositor
- ✅ Generates prompts without waiting for images

### 2. Standard (7 min, recommended)
```bash
SD_STEPS=30 \
SD_GUIDANCE=9.5 \
NUM_CANDIDATES=2 \
MAX_RETRIES=3 \
USE_CLIP=1 \
python run_e2e_test_enhanced.py
```
- ✅ Good quality/speed balance
- ✅ CLIP semantic scoring enabled
- ✅ Multi-candidate selection

### 3. High Quality (15 min)
```bash
SD_STEPS=50 \
SD_GUIDANCE=12.0 \
NUM_CANDIDATES=3 \
MAX_RETRIES=5 \
USE_CLIP=1 \
python run_e2e_test_enhanced.py
```
- ✅ Best quality output
- ✅ More candidates per panel
- ✅ More retry attempts

### 4. No LoRA/CLIP (Faster)
```bash
USE_LORA=0 \
USE_CLIP=0 \
SD_STEPS=20 \
NUM_CANDIDATES=1 \
python run_e2e_test_enhanced.py
```
- ✅ Baseline SD 1.5 only
- ✅ Sharpness-based selection only
- ✅ ~5 min runtime

### 5. Full Debugging
```bash
SD_STEPS=30 \
SD_SEED=42 \
NUM_CANDIDATES=2 \
MAX_RETRIES=3 \
USE_CLIP=1 \
DRY_RUN=0 \
python run_e2e_test_enhanced.py 2>&1 | tee pipeline.log
```
- ✅ Fixed seed for reproducibility
- ✅ Logs to file for inspection
- ✅ Full traceability

---

## Understanding the Output

### Directory Structure
```
test_output/
├── beats.json                      # Extracted narrative beats
├── layout/
│   └── layout.json                # Panel layout and positioning
├── panels/
│   └── page_chunk_001/
│       ├── panel_0.png            # Generated images
│       ├── panel_1.png
│       └── ...
├── pages/
│   └── page_chunk_001_full.png    # Composite final page
├── prompts/
│   ├── page_chunk_001_panel_0_prompt.txt
│   ├── page_chunk_001_panel_1_prompt.txt
│   └── ...
├── checkpoints/                    # Resumable checkpoints
│   ├── beats.json
│   └── layout.json
├── cache/                         # Prompt caching
└── metrics/
    └── metrics.json               # Performance metrics
```

### Metrics File (metrics.json)
```json
{
  "total_seconds": 432.5,
  "step_times": {
    "Beat Extraction": 8.2,
    "Layout Generation": 3.1,
    "Panel Generation": 410.2,
    "Compositing": 11.0
  },
  "quality": {
    "avg_score": 82.3,
    "min_score": 65.0,
    "max_score": 95.2,
    "avg_clip_score": 0.743
  },
  "reliability": {
    "total_attempts": 8,
    "generation_attempts": {
      "panel_0": 1,
      "panel_2": 2,
      "panel_4": 1
    },
    "validation_errors": 0
  },
  "memory": {
    "peak_gpu_memory_gb": 8.2,
    "peak_cpu_memory_mb": 1024
  }
}
```

### Understanding Quality Scores

**Quality Score (0-100):**
- **90-100:** Excellent (sharp, well-exposed, good contrast)
- **75-89:** Good (meets all requirements)
- **60-74:** Acceptable (minor issues)
- **<60:** Rejected (will be regenerated on retry)

**CLIP Score (0-1):**
- **0.8-1.0:** Excellent semantic match
- **0.6-0.8:** Good match
- **0.4-0.6:** Fair match
- **<0.4:** Poor match (semantic mismatch)

---

## Prompt Examples

Generated prompts are saved for transparency:

**File:** `test_output/prompts/page_chunk_001_panel_0_prompt.txt`

```
POSITIVE (72 tokens):
manga panel, medium shot of Klein (brown hair, monocle, scholarly gentleman, suit), 
discovering secret tome, dynamic dutch-angle perspective, dramatic chiaroscuro lighting, 
heavy G-pen ink shadows, fine cross-hatching on mid-tones, dense screentone on background, 
speed lines radiating from impact, sharp feathering on shadow edges, high-contrast inking, 
bold brush-pen weight variation, black and white ink, manga style, 
low-angle camera angle, rule-of-thirds composition, masterpiece, best quality, 
high resolution, detailed backgrounds, sharp focus, intricate inking details, 
professional manga, crisp lines, consistent line weight, polished artwork, 
page mood: mysterious, continuous scene, same characters: secret, discovery, Klein, 
clean manga lineart, G-pen linework, consistent character design, 
uniform inking weight, professional screentone, fine cross-hatching

NEGATIVE:
bad anatomy, distorted face, extra limbs, blurry, low quality, watermark, 
color, colored, photo, realistic, western comic, cartoon, text overlay, 
bad inking, inconsistent linework, poor quality, multiple people, group scene, crowd
```

---

## Troubleshooting

### GPU Issues

**Error: CUDA out of memory**
```bash
# Reduce inference steps
SD_STEPS=20 python run_e2e_test_enhanced.py

# Or reduce candidates
NUM_CANDIDATES=1 python run_e2e_test_enhanced.py

# Or run in CPU mode (slower but works)
CUDA_VISIBLE_DEVICES="" python run_e2e_test_enhanced.py
```

**Error: CLIP model not found**
```bash
# Install transformers
pip install transformers

# Or disable CLIP
USE_CLIP=0 python run_e2e_test_enhanced.py
```

### API Issues

**Error: GEMINI_API_KEY not set**
```bash
# Option 1: Create .env file
echo "GEMINI_API_KEY=AIza..." > .env

# Option 2: Export environment variable
export GEMINI_API_KEY="AIza..."

# Option 3: Check existing .env
cat .env
```

**Error: Failed to call Gemini API**
```bash
# Check internet connection
ping api.generativeai.google.com

# Check API key validity
# (Test with curl if needed)
```

### File Issues

**Error: Input file not found**
```bash
# Ensure test_input.txt exists in project root
ls -la test_input.txt

# Create sample if missing
echo "Once upon a time..." > test_input.txt
```

**Error: Layout validation failed**
```bash
# This is usually a warning, not an error
# Check test_output/layout/layout.json for details

# May indicate:
# - Panel count outside [4, 7]
# - Coverage outside [90%, 102%]
# - Overlapping panels
```

---

## Advanced Usage

### Custom Configuration File

Create `config.yaml`:
```yaml
canvas:
  width: 1800
  height: 2400
  panel_gap: 10
  min_panels: 4
  max_panels: 7

sd:
  steps: 30
  guidance_scale: 9.5
  max_dim: 768
  seed_base: 42
  num_candidates: 2
  max_retries: 3

pipeline:
  use_lora: true
  use_clip_scoring: true
  use_context_prompt: true
  use_continuity: true
  screentone_enabled: true
  screentone_adaptive: true

screentone:
  pattern: "dot"  # dot, line, crosshatch
  dot_radius: 2
  spacing: 6
  threshold: 128
  strength: 0.55

quality:
  quality_min_score: 60.0
  image_min_contrast: 25.0
  image_min_mean: 20.0
  image_max_mean: 235.0
```

Load with:
```python
from pathlib import Path
config = PipelineConfig.from_yaml(Path("config.yaml"))
```

### Resuming from Checkpoint

Pipeline automatically resumes from checkpoints:
```bash
# If interrupted, just re-run — will skip completed steps
python run_e2e_test_enhanced.py

# Force restart (delete checkpoints)
rm test_output/checkpoints/*.json
python run_e2e_test_enhanced.py
```

### Multi-Page Processing

```bash
# Process multiple text files sequentially
for file in test_input_*.txt; do
    echo "Processing $file..."
    python run_e2e_test_enhanced.py --input "$file"
done
```

---

## Performance Tuning

### For 12GB VRAM (RTX 3060):
```bash
SD_STEPS=25 NUM_CANDIDATES=1 MAX_RETRIES=2 USE_CLIP=0 python run_e2e_test_enhanced.py
```

### For 24GB VRAM (RTX 3090):
```bash
SD_STEPS=40 NUM_CANDIDATES=3 MAX_RETRIES=5 USE_CLIP=1 python run_e2e_test_enhanced.py
```

### For CPU-only:
```bash
CUDA_VISIBLE_DEVICES="" SD_STEPS=15 NUM_CANDIDATES=1 python run_e2e_test_enhanced.py
# Expect 30-60 min runtime
```

---

## Next Steps

1. ✅ Run basic test with DRY_RUN=1
2. ✅ Check DRY_RUN output (beats, layout)
3. ✅ Run full pipeline with standard config
4. ✅ Inspect prompts in `test_output/prompts/`
5. ✅ Review quality metrics in `metrics/metrics.json`
6. ✅ Compare output with original pipeline
7. ✅ Adjust configuration based on your results
8. ✅ Integrate improvements into production pipeline

---

## Support

- **Bugs/Issues:** Check logs in `test_output/` directory
- **CLIP Problems:** Verify with `python -c "from transformers import CLIPModel; print('OK')"`
- **Reproduction:** Include `pipeline.log` from `2>&1 | tee pipeline.log`
- **Config Questions:** See `IMPROVEMENTS_GUIDE.md` for detailed parameter explanations

---

**Happy Manga Generation! 🎨**

Last Updated: April 2026
