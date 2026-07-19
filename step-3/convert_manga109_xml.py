#!/usr/bin/env python
"""
convert_manga109_xml.py - Convert Manga109 XML annotations to JSON files for data_prep.py.

This script parses character metadata (including gender) and bounding boxes (panels, bodies, faces, texts)
from Manga109 XML and exports them as page-level JSON annotations.

Usage:
    python convert_manga109_xml.py --xml_dir C:/Manga109/annotations/ --output_dir ./step-3/annotations_json/
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_manga109_xml(xml_path: Path) -> dict:
    """Parses a single Manga109 book XML annotation file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None

    title = root.attrib.get("title", xml_path.stem)

    # 1. Parse character metadata
    characters = {}
    for char_elem in root.findall(".//character"):
        char_id = char_elem.attrib.get("id")
        if char_id:
            characters[char_id] = {
                "name": char_elem.attrib.get("name", "")
            }

    pages_data = []

    # 2. Parse pages
    for page_elem in root.findall(".//page"):
        index = int(page_elem.attrib.get("index", 0))
        width = int(page_elem.attrib.get("width", 0))
        height = int(page_elem.attrib.get("height", 0))
        page_id = f"{title}_p{index:03d}"

        # Panels
        panels = []
        for panel_elem in page_elem.findall("panel"):
            xmin = int(panel_elem.attrib.get("xmin", 0))
            ymin = int(panel_elem.attrib.get("ymin", 0))
            xmax = int(panel_elem.attrib.get("xmax", 0))
            ymax = int(panel_elem.attrib.get("ymax", 0))
            panels.append({"bbox": [xmin, ymin, xmax, ymax]})

        # Texts
        texts = []
        for text_elem in page_elem.findall("text"):
            xmin = int(text_elem.attrib.get("xmin", 0))
            ymin = int(text_elem.attrib.get("ymin", 0))
            xmax = int(text_elem.attrib.get("xmax", 0))
            ymax = int(text_elem.attrib.get("ymax", 0))
            char_id = text_elem.attrib.get("character")
            texts.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "text": text_elem.text or "",
                "character_id": char_id
            })

        # Bodies
        bodies = []
        for body_elem in page_elem.findall("body"):
            xmin = int(body_elem.attrib.get("xmin", 0))
            ymin = int(body_elem.attrib.get("ymin", 0))
            xmax = int(body_elem.attrib.get("xmax", 0))
            ymax = int(body_elem.attrib.get("ymax", 0))
            char_id = body_elem.attrib.get("character")
            bodies.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "character_id": char_id
            })

        # Faces
        faces = []
        for face_elem in page_elem.findall("face"):
            xmin = int(face_elem.attrib.get("xmin", 0))
            ymin = int(face_elem.attrib.get("ymin", 0))
            xmax = int(face_elem.attrib.get("xmax", 0))
            ymax = int(face_elem.attrib.get("ymax", 0))
            char_id = face_elem.attrib.get("character")
            faces.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "character_id": char_id
            })

        pages_data.append({
            "page_id": page_id,
            "page_number": index,
            "width": width,
            "height": height,
            "panels": panels,
            "texts": texts,
            "bodies": bodies,
            "faces": faces
        })

    return {"title": title, "pages": pages_data}

def main():
    parser = argparse.ArgumentParser(description="Convert Manga109 XML annotations to page-level JSON files")
    parser.add_argument("--xml_dir", required=True, help="Directory containing Manga109 XML files")
    parser.add_argument("--output_dir", required=True, help="Output directory to save JSON annotations")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return

    print(f"Found {len(xml_files)} XML files to convert.")

    for xml_path in xml_files:
        print(f"Processing {xml_path.name}...")
        book_data = parse_manga109_xml(xml_path)
        if not book_data:
            continue

        book_title = book_data["title"]
        book_output_dir = output_dir / book_title
        book_output_dir.mkdir(parents=True, exist_ok=True)

        for page in book_data["pages"]:
            page_id = page["page_id"]
            json_path = book_output_dir / f"{page_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page, f, indent=2, ensure_ascii=False)

    print("Conversion completed successfully.")

if __name__ == "__main__":
    main()
