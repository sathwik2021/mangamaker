"""
layout_generator.py  —  Step 2: Convert beats JSON → manga layout JSON

Usage:
    python layout_generator.py --input path/to/beats.json
    python layout_generator.py --input path/to/input_folder/
    python layout_generator.py  (uses INPUT_DIR from config.py)
"""

import os
import json
import time
import argparse
from pathlib import Path
from model_client import generate, get_current_model
from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from config import INPUT_DIR, OUTPUT_DIR, CANVAS_WIDTH, CANVAS_HEIGHT


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def load_beats_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_layout_json(data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_json_response(text: str) -> str:
    """Strip markdown code blocks if model wraps response in them."""
    if text is None or text == "":
        raise ValueError("Gemini returned empty/None response. The model may have filtered the response or hit a safety filter.")
    
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # remove first line (```json or ```) and last line (```)
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines[-1].strip() == "```" else lines
        text = "\n".join(lines)
    return text.strip()


def _append_layout_retry_reminder(prompt: str) -> str:
    """Append a retry reminder when layout validation produces warnings."""
    reminder = (
        "\n\nWhen retrying, make sure:\n"
        "1. Every beat is assigned to a panel.\n"
        "2. Bubble text stays within the allotted bubble budget.\n"
        "3. No beats are omitted or dropped.\n"
        "4. The layout remains valid JSON and follows the required panel/bubble schema.\n"
    )
    return prompt + reminder


def validate_bubble_budget(layout: dict) -> list:
    warnings = []
    for panel in layout.get("panels", []):
        bubbles = panel.get("bubbles", [])
        total_words = sum(len(b.get("text", "").split()) for b in bubbles)
        total_chars = sum(len(b.get("text", "")) for b in bubbles)
        if len(bubbles) > 2 or total_words > 25 or total_chars > 140:
            warnings.append(
                f"Panel {panel['id']}: {len(bubbles)} bubbles, "
                f"{total_words} words, {total_chars} chars — exceeds budget, may overflow bubble rendering"
            )
    return warnings


def _parse_beat_number(beat_id: str) -> int:
    try:
        return int(beat_id.split("_")[-1])
    except Exception:
        return -1


def validate_layout(layout: dict, beats_json: dict) -> list:
    """Basic validation — returns list of warnings."""
    warnings = []

    panels = layout.get("panels", [])
    if len(panels) < 4:
        warnings.append(f"Only {len(panels)} panels — expected 4-7")
    if len(panels) > 7:
        warnings.append(f"{len(panels)} panels — expected 4-7")

    W = layout.get("canvas", {}).get("width",  CANVAS_WIDTH)
    H = layout.get("canvas", {}).get("height", CANVAS_HEIGHT)

    for panel in panels:
        bbox = panel.get("bbox", [])
        if len(bbox) != 4:
            warnings.append(f"{panel['id']}: invalid bbox {bbox}")
            continue
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            warnings.append(f"{panel['id']}: bbox out of canvas bounds {bbox}")
        if x2 <= x1 or y2 <= y1:
            warnings.append(f"{panel['id']}: invalid bbox dimensions {bbox}")

        for bubble in panel.get("bubbles", []):
            bbbox = bubble.get("bbox", [])
            if len(bbbox) != 4:
                warnings.append(f"{bubble['id']}: invalid bubble bbox")
                continue
            bx1, by1, bx2, by2 = bbbox
            if bx1 < x1 or by1 < y1 or bx2 > x2 or by2 > y2:
                warnings.append(
                    f"{bubble['id']}: bubble bbox outside panel {panel['id']}"
                )

    # check all beats are covered
    input_beat_ids = {b["id"] for b in beats_json.get("beats", [])}
    used_beat_ids  = set()
    for panel in panels:
        used_beat_ids.update(panel.get("beat_ids", []))
    missing = input_beat_ids - used_beat_ids
    if missing:
        warnings.append(f"Beats not assigned to any panel: {missing}")

    # check reading_order chronologically against beats
    panel_by_id = {panel["id"]: panel for panel in panels}
    reading_order = layout.get("reading_order", [])
    if set(reading_order) != set(panel_by_id.keys()):
        warnings.append(
            "reading_order must contain every panel id exactly once in the intended narrative order"
        )
    else:
        earliest_beats = []
        for panel_id in reading_order:
            panel = panel_by_id[panel_id]
            beat_ids = panel.get("beat_ids", [])
            beat_numbers = sorted(_parse_beat_number(b) for b in beat_ids if _parse_beat_number(b) >= 0)
            earliest_beats.append(beat_numbers[0] if beat_numbers else float("inf"))

        for earlier, later, prev_id, next_id in zip(earliest_beats, earliest_beats[1:], reading_order, reading_order[1:]):
            if later < earlier:
                warnings.append(
                    "reading_order violates beat chronology: "
                    f"{next_id} appears after {prev_id} but contains earlier beats"
                )
                break

    # check bubble budget limits (FIX B)
    warnings.extend(validate_bubble_budget(layout))

    return warnings


# ─────────────────────────────────────────────────────────────
#  CORE CONVERSION
# ─────────────────────────────────────────────────────────────

def convert_beats_to_layout(beats_json: dict, max_retries: int = 3) -> dict:
    beats_str   = json.dumps(beats_json, indent=2, ensure_ascii=False)
    user_prompt = USER_PROMPT_TEMPLATE.format(beats_json=beats_str)
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    for attempt in range(1, max_retries + 1):
        print(f"  🤖 Calling {get_current_model()} (attempt {attempt}/{max_retries})...")
        try:
            raw = generate(full_prompt)
            cleaned = clean_json_response(raw)
            layout  = json.loads(cleaned)

            # inject canvas size if missing
            layout.setdefault("canvas", {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT})

            # validate
            warnings = validate_layout(layout, beats_json)
            if warnings:
                print(f"  ⚠️  Validation warnings:")
                for w in warnings:
                    print(f"     - {w}")
                if attempt < max_retries:
                    print("  🔁 Retrying with stronger bubble budget guidance...")
                    full_prompt = _append_layout_retry_reminder(full_prompt)
                    continue
                else:
                    print("  ⚠️  Final attempt completed with warnings.")
            else:
                print(f"  ✅ Layout valid.")

            return layout

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to get valid JSON after {max_retries} attempts.\n"
                    f"Last raw response:\n{raw[:500]}"
                )
            time.sleep(2 ** attempt)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  ❌ Error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


# ─────────────────────────────────────────────────────────────
#  PROCESS ONE FILE
# ─────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str):
    print(f"\n📄 Processing: {input_path}")
    beats_json = load_beats_json(input_path)
    page_id    = beats_json.get("page_id", Path(input_path).stem)
    print(f"  Page ID : {page_id}")
    print(f"  Beats   : {len(beats_json.get('beats', []))}")

    layout = convert_beats_to_layout(beats_json)
    save_layout_json(layout, output_path)
    print(f"  💾 Saved → {output_path}")
    return layout


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Step 1 beats JSON to Step 2 manga layout JSON"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to a single beats JSON file or a folder of JSON files. "
             "Defaults to INPUT_DIR in config.py"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output folder. Defaults to OUTPUT_DIR in config.py"
    )
    args = parser.parse_args()

    input_path  = args.input  or INPUT_DIR
    output_dir  = args.output or OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_path)

    # single file
    if input_path.is_file():
        output_path = Path(output_dir) / f"layout_{input_path.stem}.json"
        process_file(str(input_path), str(output_path))

    # folder of files
    elif input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            print(f"❌ No JSON files found in {input_path}")
            return
        print(f"Found {len(json_files)} JSON files to process.")
        for i, jf in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}]", end="")
            output_path = Path(output_dir) / f"layout_{jf.stem}.json"
            try:
                process_file(str(jf), str(output_path))
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue

    else:
        print(f"❌ Input path not found: {input_path}")
        return

    print(f"\n🎉 Done! Layout files saved to: {output_dir}")


if __name__ == "__main__":
    main()
