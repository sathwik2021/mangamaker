# Step 3 — Layout JSON → Manga Images

Full pipeline: beats JSON (Step 1) → layout JSON (Step 2b) → manga page images (Step 3).

## Files

| File | Purpose |
|------|---------|
| `convert_manga109_xml.py` | Step 3a: Convert Manga109 XML annotations to page-level JSON for `data_prep.py` |
| `data_prep.py` | Step 3b: Extract Manga109 panel crops + captions for training |
| `train_lora.py` | Step 3c: Train a Stable Diffusion 1.5 LoRA locally with PEFT/diffusers |
| `compositor.py` | Step 3d: Compose full manga pages from layout JSON + panel images |

## Step-by-step

### 1. Data preparation (local, no GPU)
```powershell
pip install Pillow

python convert_manga109_xml.py `
  --xml_dir "C:\...\Manga109\annotations\" `
  --output_dir "C:\...\step-3\annotations_json\"

python data_prep.py `
  --images  "C:\...\Manga109\images\" `
  --annots  "C:\...\step-3\annotations_json\" `
  --output  "C:\...\step-3\dataset\"
```

Produces:
```
dataset/
  images/          ← 512×512 panel PNGs
  captions/        ← matching .txt files
  train_metadata.jsonl
  eval_metadata.jsonl
```

> If you want subject-count tags like `1person, solo` or `2 people` in captions, run `convert_manga109_xml.py` first so `data_prep.py` can use Manga109 body/face annotations.

### Caption format note
Captions are generated using a fixed 5-mood taxonomy:
`tense`, `action`, `calm`, `emotional`, `neutral`.
The caption template is trigger-word-first and is intentionally kept in sync with the inference-side prompt builder; do not change one without checking the other.

### 2. LoRA training
This repository does not include a local `lora_train.ipynb` file. The current training path is `step-3/train_lora.py`, and the repo already contains trained weights at:

```
step-3/results/lora_output/final_lora
```

If those weights are available, retraining is optional and only needed when you want to regenerate the LoRA from scratch.

The training script defaults to the community mirror base model because `runwayml/stable-diffusion-v1-5` was deleted from Hugging Face after August 2024.

Example training command:
```powershell
python step-3/train_lora.py `
  --data_dir ./step-3/dataset `
  --output_dir ./step-3/results/lora_output/final_lora `
  --epochs 3 `
  --batch_size 2 `
  --lr 1e-4 `
  --rank 8 `
  --lora_alpha 16 `
  --mixed_precision fp16
```

Real LoRA config from `step-3/results/lora_output/final_lora/adapter_config.json`:
- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.05`
- `target_modules = ["to_q", "to_k", "to_v", "to_out.0"]`

### 3. Generate panel images (inference)
The actual production inference path in this repo is `run_e2e_test.py`, not a local notebook file.
Set these environment variables before running inference:

```powershell
$env:MANGA_LORA_PATH = "./step-3/results/lora_output/final_lora"
$env:MANGA_LORA_SCALE = "1.0"
python run_e2e_test.py
```

`run_e2e_test.py` loads the LoRA weights from `MANGA_LORA_PATH` and uses `MANGA_LORA_SCALE=1.0` by default.

### 4. Compositor (local, no GPU)
```powershell
pip install Pillow

# Single page
python step-3/compositor.py `
  --layout  "C:\...\step-2-layout\output\page_abc123.json" `
  --panels  "C:\...\step-3\panels\" `
  --output  "C:\...\step-3\output\"

# Batch (all pages)
python step-3/compositor.py `
  --layout_dir "C:\...\step-2-layout\output\" `
  --panels  "C:\...\step-3\panels\" `
  --output     "C:\...\step-3\output\" `
```

Additional compositor flags:
`--workers`, `--wobble`, `--spikes`, `--no-validate`, `--verbose`.

Output structure:
```
output/
  pages/          ← full 1800×2400 manga pages (page_id_full.png)
  panels/         ← individual panel crops (page_id/panel_N.png)
```

## Testing the compositor without trained panels

Run the compositor without `--panels` to get placeholder output:
```powershell
python step-3/compositor.py `
  --layout  "C:\...\step-2-layout\output\page_abc123.json" `
  --output  "C:\...\step-3\output\"
```
```
Panels will be rendered as gray placeholders with diagonal lines.
This lets you verify layout, borders, and bubble rendering immediately.

## Notes on Manga109 image paths

The data prep script expects:
```
images/
  AisazuNihaIrarenai/
    001.jpg
    002.jpg
    ...
  LoveHina_vol01/
    001.jpg
    ...
```

Page IDs in annotations follow the pattern `MangaTitle_pNNN` (e.g. `AisazuNihaIrarenai_p011`).
The script extracts the title by splitting on `_p`.
