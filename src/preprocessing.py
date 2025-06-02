import os
import sys
import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import List
from pathlib import Path
import pytesseract

OUTPUT_IMG_DIR = Path("output/images/")
INPUT_PDF_DIR = Path("input/")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[np.ndarray]:
    """Convert PDF to list of images (as numpy arrays)."""
    pil_images = convert_from_path(str(pdf_path), dpi=dpi)
    images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]
    return images

def correct_rotation(img: np.ndarray) -> np.ndarray:
    """Detect and correct image rotation using pytesseract's orientation detection."""
    try:
        osd = pytesseract.image_to_osd(img)
        rotation = int([line for line in osd.split('\n') if 'Rotate:' in line][0].split(':')[1].strip())
        if rotation != 0:
            # Rotate image to correct orientation
            if rotation == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rotation == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"Orientation detection failed: {e}")
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Apply orientation correction, grayscale, denoise, threshold, and deskew to the image."""
    # Correct orientation
    img = correct_rotation(img)
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Deskew
    deskewed = deskew_image(thresh)
    return deskewed

def deskew_image(img: np.ndarray) -> np.ndarray:
    """Deskew the image using its largest contour angle."""
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] == 0:
        return img  # Empty image, skip deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def save_preprocessed_images(images: List[np.ndarray], prefix: str):
    """Save only preprocessed images to output/images/ with given prefix."""
    ensure_dir(OUTPUT_IMG_DIR)
    for idx, img in enumerate(images):
        out_path = OUTPUT_IMG_DIR / f"{prefix}_page_{idx+1}_preprocessed.png"
        cv2.imwrite(str(out_path), img)

def process_pdf(pdf_path: Path):
    """Main function to process a PDF: convert to images, preprocess, and save."""
    print(f"Processing PDF: {pdf_path}")
    images = pdf_to_images(pdf_path)
    preprocessed_images = [preprocess_image(img) for img in images]
    save_preprocessed_images(preprocessed_images, pdf_path.stem)
    print(f"Saved {len(preprocessed_images)} preprocessed images to {OUTPUT_IMG_DIR.resolve()}")
    return preprocessed_images

def main():
    # For standalone testing: process all PDFs in input/
    ensure_dir(OUTPUT_IMG_DIR)
    pdf_files = list(INPUT_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_PDF_DIR.resolve()}")
        sys.exit(1)
    for pdf_path in pdf_files:
        process_pdf(pdf_path)

if __name__ == "__main__":
    main()