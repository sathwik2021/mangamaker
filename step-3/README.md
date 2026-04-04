# Step 3 — Layout JSON → Manga Images

Full pipeline: beats JSON (Step 1) → layout JSON (Step 2b) → manga page images (Step 3).

## Files

| File | Purpose |
|------|---------|
| `data_prep.py` | Step 3a: Extract Manga109 panel crops + captions for training |
| `lora_train.ipynb` | Step 3b: Kaggle notebook — fine-tune SD 1.5 LoRA on panel crops |
| `compositor.py` | Step 3e: Compose full manga pages from layout JSON + panel images |

## Step-by-step

### 1. Data preparation (local, no GPU)
```powershell
pip install Pillow

python data_prep.py `
  --images  "C:\...\Manga109\images\" `
  --annots  "C:\...\dataset-annotations-json\" `
  --output  "C:\...\step-3\dataset\"
```

Produces:
```
dataset/
  images/          ← 512×512 panel PNGs (~30K–50K images)
  captions/        ← matching .txt files
  train_metadata.jsonl
  eval_metadata.jsonl
```

### 2. Upload dataset to Kaggle
- Zip the `dataset/` folder
- Upload as a new Kaggle Dataset (e.g. "manga109-panels")

### 3. LoRA training (Kaggle, 2×T4)
- Create a new Kaggle notebook
- Attach your "manga109-panels" dataset
- Upload and run `lora_train.ipynb`
- Download `manga_lora.safetensors` from output

Expected training time: ~4–6 hours for 3 epochs on ~30K panels.

### 4. Generate panel images (inference)
Use the `generate_panels_from_layout()` function in the notebook, or run it
locally after downloading the LoRA weights.

### 5. Compositor (local, no GPU)
```powershell
pip install Pillow

# Single page
python compositor.py `
  --layout  "C:\...\step-2-layout\output\page_abc123.json" `
  --panels  "C:\...\step-3\panels\" `
  --output  "C:\...\step-3\output\"

# Batch (all pages)
python compositor.py `
  --layout_dir "C:\...\step-2-layout\output\" `
  --panels_dir "C:\...\step-3\panels\" `
  --output     "C:\...\step-3\output\"
```

Output structure:
```
output/
  pages/          ← full 1800×2400 manga pages (page_id_full.png)
  panels/         ← individual panel crops (page_id/panel_N.png)
```

## Testing the compositor without trained panels

Run the compositor without `--panels` to get placeholder output:
```powershell
python compositor.py `
  --layout  "C:\...\step-2-layout\output\page_abc123.json" `
  --output  "C:\...\step-3\output\"
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
