# data_prep.py
#
# Step 3a: Manga109 page images + annotation JSON → panel crops + captions
#
# Produces a dataset directory ready for LoRA training:
#   dataset/
#     images/   ← panel crop PNGs
#     captions/ ← matching .txt files (one caption per image)
#     metadata.jsonl ← HuggingFace-compatible metadata
#
# Usage:
#   python data_prep.py \
#     --images  C:\...\Manga109\images\ \
#     --annots  C:\...\dataset-annotations-json\ \
#     --output  C:\...\step-3\dataset\
#
# Manga109 annotation JSON structure (from your example):
#   { "page_id", "page_number", "width", "height",
#     "panels": [{"bbox": [x1,y1,x2,y2]}, ...],
#     "texts":  [{"bbox", "text", "type"}, ...] }

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MIN_PANEL_W   = 80      # discard tiny panels (gutters, slivers)
MIN_PANEL_H   = 80
MAX_PANELS    = 50000   # cap total dataset size
TRAIN_SPLIT   = 0.95    # rest goes to eval
TARGET_SIZE   = 512     # resize shorter edge to this for SD training
PADDING_COLOR = (255, 255, 255)

# ── Shot type heuristics ──────────────────────────────────────────────────────
# Approximate shot type from panel aspect ratio and area fraction of page

def _infer_shot_type(pw: int, ph: int, page_w: int, page_h: int) -> str:
    area_frac = (pw * ph) / (page_w * page_h)
    aspect    = pw / max(ph, 1)
    if area_frac > 0.30:
        return "wide"
    if aspect > 2.0:
        return "wide"
    if area_frac < 0.06:
        return "close_up"
    if aspect < 0.6:
        return "medium_close"
    return "medium"


def _infer_mood(page_id: str) -> str:
    """
    Fallback mood inference from manga title heuristics.
    Override with real beat data when available.
    Returns one of: tense, action, calm, emotional, neutral
    """
    title = page_id.lower()
    if any(k in title for k in ["fight", "battle", "gun", "war", "kung"]):
        return "action"
    if any(k in title for k in ["love", "hina", "kiss", "romance", "heart"]):
        return "emotional"  # Mapped from romantic
    if any(k in title for k in ["ghost", "horror", "dark", "demon", "youma"]):
        return "tense"
    if any(k in title for k in ["comedy", "funny", "gag", "lunch", "daily"]):
        return "calm"  # Mapped from lighthearted
    return "neutral"


# ── Caption generation ────────────────────────────────────────────────────────

def _bubble_count(panel_bbox: List[int], texts: List[Dict]) -> int:
    x1, y1, x2, y2 = panel_bbox
    count = 0
    for t in texts:
        tb = t.get("bbox", [])
        if len(tb) != 4:
            continue
        tx1, ty1, tx2, ty2 = tb
        # Check overlap with panel bbox
        if tx1 < x2 and tx2 > x1 and ty1 < y2 and ty2 > y1:
            count += 1
    return count


def _character_count(panel_bbox: List[int], bodies: List[Dict], faces: List[Dict]) -> int:
    """FIX 3b: Count character body and face bboxes whose centre falls inside
    panel_bbox. Deduplicates by character_id so a character is not double-counted.
    Returns 0 gracefully if no character annotations fall inside the panel.
    """
    char_ids = set()
    px1, py1, px2, py2 = panel_bbox

    for items in (bodies, faces):
        for item in items:
            bb = item.get("bbox", [])
            if len(bb) != 4:
                continue
            bx1, by1, bx2, by2 = bb
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                cid = item.get("character_id")
                if cid:
                    char_ids.add(cid)
    return len(char_ids)


def _generate_caption(
    panel_bbox: List[int],
    page_id: str,
    page_w: int,
    page_h: int,
    texts: List[Dict],
    panel_idx: int,
    total_panels: int,
    bodies: Optional[List[Dict]] = None,
    faces: Optional[List[Dict]] = None,
) -> str:
    x1, y1, x2, y2 = panel_bbox
    pw, ph    = x2 - x1, y2 - y1
    shot      = _infer_shot_type(pw, ph, page_w, page_h)
    mood      = _infer_mood(page_id)
    n_bubbles = _bubble_count(panel_bbox, texts)
    # FIX 3b: anchor subject count so the LoRA learns figure cardinality
    n_chars   = _character_count(panel_bbox, bodies or [], faces or [])
    subject_str = {0: "no humans, background", 1: "1person, solo"}.get(
        n_chars, f"{n_chars} people"
    )

    shot_desc = {
        "wide":         "wide establishing shot",
        "medium":       "medium shot",
        "medium_close": "medium close-up",
        "close_up":     "extreme close-up",
    }.get(shot, "manga panel")

    mood_desc = {
        "action":       "dynamic action scene",
        "emotional":    "emotional moment",
        "tense":        "tense dramatic moment",
        "calm":         "calm scene",
        "neutral":      "scene",
    }.get(mood, "scene")

    bubble_str = ""
    if n_bubbles == 1:
        bubble_str = ", one speech bubble"
    elif n_bubbles > 1:
        bubble_str = f", {n_bubbles} speech bubbles"

    position_str = ""
    if total_panels > 1:
        frac = panel_idx / (total_panels - 1) if total_panels > 1 else 0
        if frac < 0.25:
            position_str = ", opening panel"
        elif frac > 0.75:
            position_str = ", closing panel"

    # FIX 3: Start with trigger words; FIX 3b: inject subject count
    caption = (
        f"manga style, monochrome, {subject_str}, {shot_desc}, "
        f"{mood_desc}{bubble_str}{position_str}, "
        f"screentone, hatching, fine ink lines, masterpiece, best quality"
    )
    return caption


# ── Panel extraction ──────────────────────────────────────────────────────────

def _process_annotation(
    annot_path: Path,
    images_root: Path,
    output_dir: Path,
    records: List[Dict],
) -> int:
    with open(annot_path, encoding="utf-8") as f:
        annot = json.load(f)

    page_id  = annot.get("page_id", annot_path.stem)
    page_num = annot.get("page_number", 0)
    page_w   = annot.get("width",  1654)
    page_h   = annot.get("height", 1170)
    panels   = annot.get("panels", [])
    texts    = annot.get("texts",  [])
    bodies   = annot.get("bodies", [])
    faces    = annot.get("faces", [])

    # Find matching image
    # Manga109 structure: images/<MangaTitle>/<NNN>.jpg
    manga_title = page_id.rsplit("_p", 1)[0] if "_p" in page_id else page_id
    img_filename = f"{page_num:03d}.jpg"
    img_path = images_root / manga_title / img_filename

    if not img_path.exists():
        # Try alternate naming conventions
        for ext in [".jpg", ".png", ".JPG", ".PNG"]:
            alt = images_root / manga_title / f"{page_num:03d}{ext}"
            if alt.exists():
                img_path = alt
                break
        else:
            return 0

    try:
        page_img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"  Cannot open {img_path}: {exc}")
        return 0

    # Use actual image dimensions if annotation differs
    actual_w, actual_h = page_img.size
    scale_x = actual_w / page_w
    scale_y = actual_h / page_h

    saved = 0
    for idx, panel in enumerate(panels):
        bbox = panel.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Scale to actual image coordinates
        x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

        pw, ph = x2 - x1, y2 - y1
        if pw < MIN_PANEL_W or ph < MIN_PANEL_H:
            continue

        # Crop
        crop = page_img.crop((x1, y1, x2, y2))

        # Resize: shorter edge → TARGET_SIZE, pad to square
        ratio  = TARGET_SIZE / min(pw, ph)
        new_w  = int(pw * ratio)
        new_h  = int(ph * ratio)
        resized = crop.resize((new_w, new_h), Image.LANCZOS)

        # Pad to TARGET_SIZE × TARGET_SIZE
        padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), PADDING_COLOR)
        off_x  = (TARGET_SIZE - new_w) // 2
        off_y  = (TARGET_SIZE - new_h) // 2
        # If resized is larger than target on one dimension, crop center
        if new_w > TARGET_SIZE or new_h > TARGET_SIZE:
            resized = resized.crop((
                max(0, (new_w - TARGET_SIZE) // 2),
                max(0, (new_h - TARGET_SIZE) // 2),
                max(0, (new_w - TARGET_SIZE) // 2) + TARGET_SIZE,
                max(0, (new_h - TARGET_SIZE) // 2) + TARGET_SIZE,
            ))
            padded.paste(resized, (0, 0))
        else:
            padded.paste(resized, (off_x, off_y))

        # File names
        stem    = f"{page_id}_panel{idx:03d}"
        img_out = output_dir / "images" / f"{stem}.png"
        cap_out = output_dir / "captions" / f"{stem}.txt"

        padded.save(img_out, "PNG")

        caption = _generate_caption(
            [x1, y1, x2, y2], page_id, actual_w, actual_h,
            texts, idx, len(panels),
            bodies=bodies,
            faces=faces,
        )
        cap_out.write_text(caption, encoding="utf-8")

        records.append({
            "file_name": f"images/{stem}.png",
            "text":      caption,
            "page_id":   page_id,
            "panel_idx": idx,
        })
        saved += 1

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3a — extract Manga109 panel crops for LoRA training"
    )
    parser.add_argument("--images",  required=True,
                        help="Manga109 images root (contains per-title folders)")
    parser.add_argument("--annots",  required=True,
                        help="Annotation JSON directory")
    parser.add_argument("--output",  required=True,
                        help="Output dataset directory")
    parser.add_argument("--max_panels", type=int, default=MAX_PANELS,
                        help=f"Max total panels to extract (default {MAX_PANELS})")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir  = Path(args.output)
    images_root = Path(args.images)
    annots_dir  = Path(args.annots)

    (output_dir / "images").mkdir(parents=True,   exist_ok=True)
    (output_dir / "captions").mkdir(parents=True, exist_ok=True)

    annot_files = sorted(annots_dir.glob("**/*.json"))
    print(f"Found {len(annot_files)} annotation files")

    records: List[Dict] = []
    total_saved = 0

    for i, af in enumerate(annot_files, 1):
        if total_saved >= args.max_panels:
            print(f"Reached max_panels={args.max_panels}, stopping.")
            break
        print(f"[{i}/{len(annot_files)}] {af.name}  ({total_saved} panels so far)")
        n = _process_annotation(af, images_root, output_dir, records)
        total_saved += n

    # Shuffle and split
    random.shuffle(records)
    split     = int(len(records) * TRAIN_SPLIT)
    train_recs = records[:split]
    eval_recs  = records[split:]

    # Write HuggingFace metadata.jsonl
    meta_train = output_dir / "train_metadata.jsonl"
    meta_eval  = output_dir / "eval_metadata.jsonl"

    with open(meta_train, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(meta_eval, "w", encoding="utf-8") as f:
        for r in eval_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {total_saved} panels extracted.")
    print(f"  Train: {len(train_recs)}  |  Eval: {len(eval_recs)}")
    print(f"  Output: {output_dir}")
    print(f"\nNext step: upload {output_dir} to Kaggle and run lora_train.ipynb")


if __name__ == "__main__":
    main()
