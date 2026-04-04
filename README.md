# AI Manga Generation Pipeline

An end-to-end, GPU-accelerated manga generation pipeline that creates fully structured manga pages from raw text. This pipeline reads a narrative, extracts visual beats, structures a comic layout, generates manga-styled imagery with Stable Diffusion, and composites the final page layout onto a blank canvas—all through a beautiful local Web Interface!

## 🚀 Features
- **Narrative Beat Extraction:** Uses an LLM (Gemini) to parse unstructured text into visual scenes (beats).
- **Automated Layout Generation:** Dynamically builds a comic layout constraint layout with varying panel sizes and bounding boxes based on the story's rhythm.
- **GPU-Accelerated Inference:** Generates detailed manga imagery using Stable Diffusion optimizations and custom LoRAs via `diffusers` & `peft`.
- **Image Compositing:** Cropping, resizing, and smart compositing of generated images into their respective manga frames using OpenCV.
- **Responsive Web UI:** Full Flask-based web application with progress bars, parallel job queuing, and interactive results downloading.

---

## 💻 Web Interface Quick Start

### 1. Environment Setup

Ensure you are using Python 3.12 and have a CUDA-compatible environment. 
Open a terminal in the root folder and activate the environment:

```powershell
.\venv_gpu\Scripts\Activate.ps1
```

*(Note: The environment requires `torch+cu121`, `transformers`, `diffusers`, `opencv-python`, and `peft`. Avoid pinning isolated package upgrades as `diffusers` relies heavily on exact `peft` and `transformers` pairings.)*

### 2. Configure API Keys

The pipeline relies on the Gemini API for the Beat Extraction module. Set the key locally securely:

```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

*(Warning: Never commit your `.env` file or export your API key publicly!)*

### 3. Launch the Server

Use the provided PowerShell launcher to boot the Flask application securely:

```powershell
.\start_web.ps1
```

*(If you get script execution permission errors, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 4. Open the App
Visit http://localhost:5000 in your browser! 
- Input a short story or narrative text.
- Configure Image Steps and Max Beats.
- Click **Generate Manga** and watch the real-time layout creation.
- Once finished, you can preview and download the finished PNG page right from the dashboard.

---

## 📂 Architecture Layout

- `app.py` — The core Web Application layer queuing async pipeline tasks.
- `/static` & `/templates` — Browser-side JavaScript tracking progress and HTML UI logic.
- `/Step-1/code/` — **Extraction Module.** NLP logic converting sentences into `[id, description, mood]` JSON dictionaries.
- `/step-2-layout/` — **Draft Module.** Analyzes JSON beats to create optimal 4-7 panel bounding boxes.
- `/step-3/` — **Compositing Module.** Takes the bounds, prompts Stable Diffusion via GPU, and crops/pastes output streams into a singular cohesive page.
- `/web_output/` — Cache containing all temporary JSON layouts and successfully generated manga `.png` results.

---

## 🛠️ Troubleshooting

**"Failed to extract beats: Beat extraction module not available"**
- Your Python environment failed to import dependencies during startup. Check if you recently downgraded `diffusers`, `transformers`, `peft` or missed `opencv-python`. Run `python test_imports.py` to identify the missing package.

**"File Not Found" on Web Download**
- Refresh your UI. Previous jobs fetched before the `app.js` local-routing patch may have cached an outdated path string. 

**"String indices must be integers, not 'str'"**
- The LLM failed to return a proper JSON dictionary for beats. The pipeline enforces a strict schema, but on rare API hiccups, Gemini might hallucinate markdown. Try pushing "Generate" again or tweak your prompt.

## 🤝 Project State
- Environment: Completely configured with isolated Git exclusions.
- Status: **Fully Operational locally.**
