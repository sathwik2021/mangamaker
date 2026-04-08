# AI Manga Generation Pipeline

> An end-to-end, GPU-accelerated pipeline that transforms raw narrative text into fully structured, publication-ready manga pages — powered by Google Gemini, Stable Diffusion 1.5, and a custom LoRA fine-tuned on the Manga109 dataset.

---

## 📖 Overview

The AI Manga Generation Pipeline bridges the gap between unstructured literary text and the precise visual grammar of Japanese manga. The system integrates Google Gemini LLMs for narrative understanding and a LoRA-fine-tuned Stable Diffusion 1.5 model for high-fidelity manga-style image synthesis, all orchestrated through a responsive Flask web interface.

---

## ✨ Features

- **Narrative Beat Extraction** — Uses the Google Gemini API to parse unstructured prose into structured visual scenes (`[id, description, mood, intensity]` JSON dictionaries).
- **Rule-Constrained Layout Generation** — Dynamically builds manga-compliant panel bounding boxes, shot types, speech bubble coordinates, and cinematic compositions using a 20-rule constraint system.
- **LoRA-Fine-Tuned Image Synthesis** — Generates authentic monochrome manga imagery via a custom LoRA trained on the [Manga109 dataset (Hugging Face)](https://huggingface.co/datasets/manga109), fine-tuned at 15,500 steps on a Kaggle T4 GPU environment.
- **Intelligent Panel Compositing** — Performs aspect-ratio-aware cropping, adaptive screentone rendering, speech bubble drawing, and full-page assembly at 1800 × 2400 px using OpenCV and Pillow.
- **Responsive Web Interface** — Flask-based single-page application with real-time progress tracking via SSE, parallel job queuing, and one-click PNG download.

---

## 🗂️ Dataset

The image generation model was fine-tuned on the **[Manga109 dataset](https://huggingface.co/datasets/manga109)**, sourced directly from Hugging Face (`manga109/Manga109`).

| Attribute | Value |
|---|---|
| Total Manga Volumes | 109 |
| Total Pages | ~21,000 |
| Panel Crops (Training Set) | ~30,000 – 50,000 |
| Image Resolution | 512 × 512 px |
| Caption Format | Booru-tag style |
| Train / Val / Test Split | 80% / 10% / 10% |
| Annotation Format | XML (frame, character, text, face bounding boxes) |

> **Citation:** Matsui, Y., et al. (2017). *Sketch-based Manga Retrieval using Manga109 Dataset.* Multimedia Tools and Applications, 76(20), 21811–21838.

---

## 💻 Quick Start

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (6 GB+ VRAM recommended)
- `torch+cu121`, `diffusers`, `transformers`, `peft`, `opencv-python`

> ⚠️ Avoid pinning isolated package upgrades — `diffusers` relies on exact `peft` and `transformers` version pairings.

---

### 1. Activate the Environment

```powershell
.\venv_gpu\Scripts\Activate.ps1
```

> If you encounter script execution permission errors, run the following first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

---

### 2. Configure API Keys

The pipeline uses the Gemini API for Beat Extraction and Layout Generation. Set your key as a local environment variable:

```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

> ⚠️ **Never commit your `.env` file or expose your API key in version control.**

---

### 3. Launch the Server

```powershell
.\start_web.ps1
```

---

### 4. Open the App

Visit **http://localhost:5000** in your browser.

1. Paste a short story or narrative text into the input area.
2. Configure **Image Steps** and **Max Beats** using the sliders.
3. Click **Generate Manga** and monitor real-time progress.
4. Preview and download the finished manga page as a PNG.

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
| `/web_output/` | Output cache — temporary layout JSON files and generated manga PNG results |

---

## ⚙️ LoRA Training Configuration

The custom LoRA was trained on Kaggle's dual-T4 GPU infrastructure over three resumable sessions (~4–6 hours each).

| Hyperparameter | Value |
|---|---|
| Base Model | `runwayml/stable-diffusion-v1-5` |
| Training Steps | 15,500 |
| Learning Rate | 1e-4 with cosine annealing |
| Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Mixed Precision | fp16 |
| LoRA Rank | 4 |
| Optimizer | AdamW (weight decay = 1e-2) |
| Hardware | Kaggle Dual T4 GPU (2 × 16 GB VRAM) |

---

## 📦 Dependencies

```
torch==2.5.1+cu121       # GPU-accelerated tensor computation (CUDA 12.1)
diffusers>=0.37.1        # Stable Diffusion inference pipeline
peft>=0.18.1             # LoRA adapter loading and management
transformers>=4.47.0     # CLIP text encoder and HuggingFace model hub
accelerate>=1.13.0       # Mixed-precision and device management
safetensors>=0.7.0       # Efficient LoRA weight serialization
google-genai             # Gemini API client (beat extraction and layout generation)
flask>=3.0.0             # Web application framework
opencv-python            # Image processing and compositing
scipy                    # Signal processing for screentone rendering
pillow                   # Image I/O and drawing (speech bubbles)
numpy                    # Array operations
python-dotenv>=1.2.2     # Environment variable management
```

---

## 🛠️ Troubleshooting

**`"Failed to extract beats: Beat extraction module not available"`**

The Python environment failed to import dependencies at startup. This usually means a missing or mismatched package. Verify with:
```bash
python test_imports.py
```
Check for recent downgrades to `diffusers`, `transformers`, `peft`, or a missing `opencv-python` installation.

---

**`"File Not Found"` on web download**

Refresh the UI. Jobs fetched before the `app.js` local-routing patch may have cached an outdated output path string.

---

**`"String indices must be integers, not 'str'"`**

The Gemini API returned an improperly structured JSON response for the beat extraction step. The pipeline enforces a strict schema, but on rare API hiccups the model may include markdown fencing in its output. Try clicking **Generate** again or slightly rephrase your input prompt.

---

## 📊 Project Status

| Component | Status |
|---|---|
| Python Environment | ✅ Fully configured with isolated Git exclusions |
| Beat Extraction (Step 1) | ✅ Operational |
| Layout Generation (Step 2) | ✅ Operational |
| LoRA Training | ✅ Complete — checkpoint-15500 |
| Image Generation & Compositing (Step 3) | ✅ Operational |
| Flask Web Interface | ✅ Operational |
| Overall System | ✅ **Fully Operational Locally** |
