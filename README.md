# AI Manga Generation Pipeline

> An end-to-end, GPU-accelerated pipeline that transforms raw narrative text into fully structured, publication-ready manga pages — powered by Google Gemini, Stable Diffusion 1.5, and a custom LoRA fine-tuned on the Manga109 dataset.

---

## 📖 Overview

The AI Manga Generation Pipeline bridges the gap between unstructured literary text and the precise visual grammar of Japanese manga. The system integrates Google Gemini LLMs for narrative understanding and a LoRA-fine-tuned Stable Diffusion 1.5 model for high-fidelity manga-style image synthesis, all orchestrated through a responsive Flask web interface.

---

## ✨ Features

- **Narrative Beat Extraction** — Uses the Google Gemini API to parse unstructured prose into structured visual scenes (`[id, description, mood, intensity]` JSON dictionaries).
- **Rule-Constrained Layout Generation** — Dynamically builds manga-compliant panel bounding boxes, shot types, speech bubble coordinates, and cinematic compositions using a 20-rule constraint system covering reading order, gutter spacing, and balloon placement.
- **LoRA-Fine-Tuned Image Synthesis** — Generates authentic monochrome manga imagery via a custom LoRA trained on the [Manga109 dataset (Hugging Face)](https://huggingface.co/datasets/manga109), fine-tuned at 15,500 steps on a Kaggle T4 GPU environment.
- **Intelligent Panel Compositing** — Performs aspect-ratio-aware cropping, adaptive screentone rendering, speech bubble drawing, and full-page assembly at 1800 × 2400 px using OpenCV and Pillow.
- **Responsive Web Interface** — Flask-based single-page application with real-time progress tracking via SSE, job status polling, and one-click PNG download.
- **Measured performance**: ~30 seconds per panel on an NVIDIA RTX 3050 (6GB), GPU-verified via CUDA profiling (excludes one-time model download).

---

## 🗂️ Dataset

The image generation model was fine-tuned on the **[Manga109 dataset](https://huggingface.co/datasets/manga109)**, sourced directly from Hugging Face (`manga109/Manga109`).

Manga109 was created by the **Aizawa Yamasaki Matsui Laboratory, The University of Tokyo**, and is provided strictly for **academic research purposes**. The manga pages themselves remain copyrighted by their original authors and publishers; Manga109 provides them under special permission for non-commercial research use only. This project uses the dataset accordingly — for academic/personal learning purposes — and does not redistribute the original manga images or use them for commercial purposes.

| Attribute | Value |
|---|---|
| Source Institution | The University of Tokyo (Aizawa Yamasaki Matsui Lab) |
| Total Manga Volumes | 109 |
| Total Pages | ~21,000 |
| Panel Crops (Training Set) | ~30,000 – 50,000 |
| Image Resolution | 512 × 512 px |
| Caption Format | Booru-tag style |
| Train / Val / Test Split | 80% / 10% / 10% |
| Annotation Format | XML (frame, character, text, face bounding boxes) |
| License | Academic research use only — see [Manga109 official site](http://www.manga109.org/) for full terms |

> **Citation:** Matsui, Y., et al. (2017). *Sketch-based Manga Retrieval using Manga109 Dataset.* Multimedia Tools and Applications, 76(20), 21811–21838.

---

## 💻 Quick Start

### Prerequisites

- **Python 3.12 specifically** — `torch+cu121` does not have a wheel for Python 3.13 at the time of writing. If you have a newer Python installed system-wide, use conda (below) rather than fighting version mismatches.
- CUDA-compatible NVIDIA GPU (6 GB+ VRAM recommended — tested working on RTX 3050 6GB)
- Conda (Miniconda or Anaconda) — recommended over plain `venv` since it can install an isolated Python 3.12 regardless of your system's default Python version.

> ⚠️ Avoid pinning isolated package upgrades — `diffusers` relies on exact `peft` and `transformers` version pairings. **`transformers` must stay below version 5.0** — a major version bump silently changes the default CLIP text encoder config and causes shape-mismatch errors when loading Stable Diffusion 1.5 weights.

---

### 1. Create the environment (conda recommended)

```bash
conda create -n mangamaker python=3.12 -y
conda activate mangamaker
```

### 2. Install PyTorch with CUDA support

Install this **before** the rest of requirements.txt, since it needs a special index:

```bash
conda install pytorch==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Verify your GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
This should print `True` followed by your GPU name. If it prints `False`, do not proceed — image generation will silently fall back to CPU and be extremely slow.

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify imports

```bash
python tests/test_imports.py
```
All packages should report `OK`. If `torchvision` warnings appear regarding `CLIPImageProcessor`, install it explicitly (matching your torch version):
```bash
pip install torchvision==0.20.1
```

### 5. Configure API Keys

Create a `.env` file in the project root (never commit this file):
```
GEMINI_API_KEY=your-api-key-here
```

### 6. (Windows only) Avoid symlink cache corruption

Windows blocks HuggingFace's default symlink caching unless Developer Mode or admin rights are enabled, which can leave partially-downloaded models in a broken state (missing `config.json` files). Add this to your `.env` as a precaution:
```
HF_HUB_DISABLE_SYMLINKS=1
```

### 7. Launch the Server

```bash
python app.py
```
(The included `start_web.ps1`/`start_web.bat` scripts assume a `venv_gpu` folder from an older setup approach — if you followed the conda steps above, just run `python app.py` directly instead.)

### 8. Open the App

Visit **http://localhost:5000** in your browser.

1. Paste a short story or narrative text into the input area.
2. Configure **Image Steps** and **Max Beats** using the sliders.
3. Click **Generate Manga** and monitor real-time progress.
4. Preview and download the finished manga page as a PNG.

**First generation will take significantly longer** (~10 minutes) since it downloads the ~4.3GB Stable Diffusion 1.5 base model. Subsequent generations run in roughly 3 minutes for a 4-panel page (~30s/panel).

---

## 🏗️ Architecture

```
Raw Text
   │
   ▼
[Step 1] Beat Extraction       ← Gemini API → structured Beat JSON
   │
   ▼
[Step 2] Layout Generation     ← Gemini API (20-rule system) → panel bounding boxes + Layout JSON
   │
   ▼
[Step 3] Image Generation      ← SD 1.5 + Manga109 LoRA → panel PNGs
         + Compositing         ← OpenCV + Pillow → final 1800×2400 page PNG
   │
   ▼
[Flask Web App]                ← SSE progress events → browser preview + download
```

### Directory Structure

| Path | Description |
|---|---|
| `app.py` | Flask web application — job queuing, routing, SSE progress events |
| `/static` & `/templates` | Frontend SPA — HTML, CSS, JavaScript |
| `/Step-1/code/` | Beat Extraction Module — NLP logic converting prose into `[id, description, mood]` JSON |
| `/step-2-layout/` | Layout Generation Module — 20-rule constraint system for panel bounding boxes |
| `/step-3/` | Image Generation & Compositing Module — SD 1.5 + LoRA inference, screentone rendering, page assembly |
| `/tests/` | Test/debug scripts, including `run_e2e_test.py` (exports `step1_extract_beats`, used by `app.py`) |
| `/docs/` | Planning documents (implementation plan, pipeline specification) |
| `/web_output/` | Output cache — temporary layout JSON files and generated manga PNG results |

---

## ⚙️ LoRA Training Configuration

The custom LoRA was trained on Kaggle's dual-T4 GPU infrastructure over three resumable sessions (~4–6 hours each).

| Hyperparameter | Value |
|---|---|
| Base Model | Stable Diffusion 1.5 (see note below on model source) |
| Training Steps | 15,500 |
| Learning Rate | 1e-4 with cosine annealing |
| Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Mixed Precision | fp16 |
| LoRA Rank | 4 |
| Optimizer | AdamW (weight decay = 1e-2) |
| Hardware | Kaggle Dual T4 GPU (2 × 16 GB VRAM) |

> **Note on base model source:** `runwayml/stable-diffusion-v1-5` was removed from Hugging Face in 2024. The pipeline now loads from the maintained community mirror `stable-diffusion-v1-5/stable-diffusion-v1-5`, which redirects automatically in most cases but can leave a corrupted local cache on Windows (see Troubleshooting).

---

## 📦 Dependencies

See `requirements.txt` for the full pinned list. Key packages:

```
torch==2.5.1                   # install via conda with pytorch-cuda=12.1, not pip
diffusers>=0.37.1              # Stable Diffusion inference pipeline
peft>=0.18.1                   # LoRA adapter loading and management
transformers>=4.47.0,<5.0.0    # CLIP text encoder — must stay below v5 (see Troubleshooting)
accelerate>=1.13.0             # Mixed-precision and device management
safetensors>=0.7.0             # Efficient LoRA weight serialization
google-genai                   # Gemini API client (beat extraction and layout generation)
flask>=3.0.0                   # Web application framework
flask-cors>=4.0.0              # CORS support for the web API
opencv-python                  # Image processing and compositing
scipy                          # Signal processing for screentone rendering
pillow                         # Image I/O and drawing (speech bubbles)
numpy                          # Array operations
python-dotenv>=1.2.2           # Environment variable management
```

---

## 🛠️ Troubleshooting

**`"Failed to extract beats: Beat extraction module not available"`**

The Python environment failed to import dependencies at startup, or a required module isn't on `sys.path`. Verify with:
```bash
python tests/test_imports.py
```
Check for recent version drift in `diffusers`, `transformers`, `peft`, or a missing `opencv-python` installation.

---

**`OSError: Error no file named config.json found in directory ...\vae` (or `...\unet`)**

This means the local Hugging Face model cache is corrupted or incomplete — commonly caused by Windows blocking symlinks during download (see the "Developer Mode" warning in the console). Fix:
```bash
rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-v1-5"
set HF_HUB_DISABLE_SYMLINKS=1
python app.py
```
This forces a clean re-download using real file copies instead of symlinks.

---

**`RuntimeError: You set ignore_mismatched_sizes to False, thus raising an error` (CLIP text encoder shape mismatch, e.g. 768 vs 512)**

This means `transformers` was upgraded to version 5.x, which changed the default CLIP text encoder configuration and breaks loading of Stable Diffusion 1.5's original weights. Fix:
```bash
pip install "transformers>=4.47.0,<5.0.0"
```

---

**`ModuleNotFoundError: No module named 'compositor'`**

`step-3/` (where `compositor.py` lives) wasn't added to `sys.path` in `app.py`. Ensure `app.py` includes:
```python
sys.path.insert(0, str(STEP3_DIR))
```
alongside the other `sys.path.insert()` calls for Step 1 and Step 2.

---

**`UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f916'`**

Windows' default console encoding can't display emoji used in some log/print statements. Fix:
```bash
set PYTHONIOENCODING=utf-8
python app.py
```
Or add this near the top of `app.py`:
```python
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
```

---

**`"File Not Found"` on web download**

Refresh the UI. Jobs fetched before the `app.js` local-routing patch may have cached an outdated output path string.

---

**`"String indices must be integers, not 'str'"`**

The Gemini API returned an improperly structured JSON response for the beat extraction step. The pipeline enforces a strict schema, but on rare API hiccups the model may include markdown fencing in its output. Try clicking **Generate** again or slightly rephrase your input prompt.

---

**Prompts truncated: `"CLIP can only handle sequences up to 77 tokens"`**

CLIP's text encoder has a hard 77-token input limit. Long, tag-heavy prompts (common with detailed style tags) get silently truncated, which can affect image quality. This is a known limitation — a production fix would involve prompt chunking or a manual token budget rather than relying on the full generated prompt string.

---

## 📊 Project Status

| Component | Status |
|---|---|
| Python Environment | ✅ Fully configured with isolated Git exclusions (conda, Python 3.12) |
| Beat Extraction (Step 1) | ✅ Operational |
| Layout Generation (Step 2) | ✅ Operational |
| LoRA Training | ✅ Complete — checkpoint-15500 |
| Image Generation & Compositing (Step 3) | ✅ Operational — GPU-verified on RTX 3050 |
| Flask Web Interface | ✅ Operational |
| Overall System | ✅ **Fully Operational Locally** |

### Known Limitations
- CLIP's 77-token limit truncates long style-tag prompts, potentially reducing generation fidelity for complex scenes.
- No automated tests for the layout generation's 20-rule constraint system.
- Base Stable Diffusion 1.5 model source changed from `runwayml/stable-diffusion-v1-5` (deprecated) to the community-maintained mirror; pipeline updated accordingly.