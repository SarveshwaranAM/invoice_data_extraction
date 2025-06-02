import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

PREPROCESSED_IMG_DIR = Path("output/images/")
SIGNATURE_OUTPUT_DIR = Path("output/signatures/")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def detect_signature_region(img: np.ndarray, crop_height_ratio: float = 0.18, crop_width_ratio: float = 0.35) -> np.ndarray:
    """
    Focus on the bottom-right region of the image for signature/seal detection.
    Returns the cropped signature image or None if not found.
    """
    h, w = img.shape[:2]
    # Crop bottom-right region
    crop_y = int(h * (1 - crop_height_ratio))
    crop_x = int(w * (1 - crop_width_ratio))
    region = img[crop_y:h, crop_x:w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle varying backgrounds
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area and aspect ratio
    candidates = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        aspect = w_box / (h_box + 1e-5)
        # Area and aspect ratio tuned for typical signature size
        if 800 < area < 20000 and 1.5 < aspect < 8.0:
            # Ignore contours touching the left/bottom edge (likely black border)
            if x > 5 and y + h_box < region.shape[0] - 5:
                candidates.append((area, x, y, w_box, h_box))
    if candidates:
        # Pick the largest candidate
        _, x, y, w_box, h_box = max(candidates, key=lambda tup: tup[0])
        signature_crop = region[y:y+h_box, x:x+w_box]
        return signature_crop
    else:
        return None

def process_signature_for_image(image_path: Path, prefix: str, page_num: int) -> Dict[str, Any]:
    img = cv2.imread(str(image_path))
    signature_img = detect_signature_region(img)
    ensure_dir(SIGNATURE_OUTPUT_DIR)
    out_path = SIGNATURE_OUTPUT_DIR / f"{prefix}_page_{page_num}_signature.png"
    if signature_img is not None:
        cv2.imwrite(str(out_path), signature_img)
        seal_present = True
        signature_image_path = str(out_path)
    else:
        seal_present = False
        signature_image_path = None
    return {
        "seal_present": seal_present,
        "signature_image": signature_image_path
    }

def process_all_images(prefix: str):
    image_files = sorted(PREPROCESSED_IMG_DIR.glob(f"{prefix}_page_*_preprocessed.png"))
    if not image_files:
        print(f"No preprocessed images found for prefix '{prefix}' in {PREPROCESSED_IMG_DIR.resolve()}")
        sys.exit(1)
    results = []
    for idx, img_path in enumerate(image_files):
        print(f"Detecting signature/seal in {img_path.name} ...")
        result = process_signature_for_image(img_path, prefix, idx + 1)
        results.append(result)
    # Save summary JSON
    summary_path = SIGNATURE_OUTPUT_DIR / f"{prefix}_signature_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"Signature detection results saved to {summary_path.resolve()}")
    return results

def main():
    # For standalone testing: process all preprocessed images for all prefixes
    ensure_dir(SIGNATURE_OUTPUT_DIR)
    all_imgs = list(PREPROCESSED_IMG_DIR.glob("*_preprocessed.png"))
    prefixes = sorted(set("_".join(img.name.split("_")[:-3]) for img in all_imgs))
    if not prefixes:
        print(f"No preprocessed images found in {PREPROCESSED_IMG_DIR.resolve()}")
        sys.exit(1)
    for prefix in prefixes:
        process_all_images(prefix)

if __name__ == "__main__":
    main()