import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import pytesseract
import numpy as np

PREPROCESSED_IMG_DIR = Path("output/images/")
OCR_OUTPUT_DIR = Path("output/ocr/")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_preprocessed_images(prefix: str) -> List[Path]:
    """Get all preprocessed images for a given PDF prefix."""
    files = sorted(PREPROCESSED_IMG_DIR.glob(f"{prefix}_page_*_preprocessed.png"))
    return files

def run_ocr_on_image(image_path: Path) -> List[Dict[str, Any]]:
    """Run pytesseract OCR on an image and return word-level data."""
    img = cv2.imread(str(image_path))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if data['text'][i].strip() == "":
            continue
        word_info = {
            "text": data['text'][i],
            "left": int(data['left'][i]),
            "top": int(data['top'][i]),
            "width": int(data['width'][i]),
            "height": int(data['height'][i]),
            "conf": float(data['conf'][i])
        }
        words.append(word_info)
    return words

def save_ocr_output(words: List[Dict[str, Any]], page_num: int, prefix: str):
    """Save OCR output as JSON."""
    ensure_dir(OCR_OUTPUT_DIR)
    out_path = OCR_OUTPUT_DIR / f"{prefix}_page_{page_num}_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2, ensure_ascii=False)

def process_all_images(prefix: str):
    """Process all preprocessed images for a given PDF prefix."""
    image_files = get_preprocessed_images(prefix)
    if not image_files:
        print(f"No preprocessed images found for prefix '{prefix}' in {PREPROCESSED_IMG_DIR.resolve()}")
        sys.exit(1)
    for idx, img_path in enumerate(image_files):
        print(f"OCR on {img_path.name} ...")
        words = run_ocr_on_image(img_path)
        save_ocr_output(words, idx + 1, prefix)
    print(f"OCR results saved to {OCR_OUTPUT_DIR.resolve()}")

def main():
    # For standalone testing: process all preprocessed images for all prefixes
    ensure_dir(OCR_OUTPUT_DIR)
    # Find all unique prefixes in preprocessed images
    all_imgs = list(PREPROCESSED_IMG_DIR.glob("*_preprocessed.png"))
    prefixes = sorted(set("_".join(img.name.split("_")[:-3]) for img in all_imgs))
    if not prefixes:
        print(f"No preprocessed images found in {PREPROCESSED_IMG_DIR.resolve()}")
        sys.exit(1)
    for prefix in prefixes:
        process_all_images(prefix)

if __name__ == "__main__":
    main()