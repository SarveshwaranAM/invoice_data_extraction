import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

FIELDS_OUTPUT_DIR = Path("output/")
PREPROCESSED_IMG_DIR = Path("output/images/")
OCR_OUTPUT_DIR = Path("output/ocr/")
LINEITEMS_OUTPUT_DIR = Path("output/")
VERIFICATION_OUTPUT_DIR = Path("output/")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ocr_json(page_json_path: Path) -> List[Dict[str, Any]]:
    with open(page_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_calculations(fields: Dict[str, Any], line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Extract values
    try:
        subtotal = sum(float(item.get("row_total", 0)) for item in line_items)
        extracted_subtotal = float(fields.get("subtotal", {}).get("value", 0))
        gst = float(fields.get("gst_amount", {}).get("value", 0))
        discount = float(fields.get("discount", {}).get("value", 0))
        total = float(fields.get("total", {}).get("value", 0))
    except Exception as e:
        return {
            "verified": False,
            "error": f"Value extraction error: {e}",
            "confidence": 0.0,
            "error_margin": None
        }

    # Compute expected total
    computed_total = subtotal + gst - discount
    error_margin = abs(computed_total - total)
    verified = error_margin < 1.0  # Acceptable error margin (tune as needed)
    confidence = max(0.0, 1.0 - (error_margin / (total + 1e-5)))

    return {
        "verified": verified,
        "confidence": round(confidence, 3),
        "error_margin": round(error_margin, 2),
        "extracted_subtotal": extracted_subtotal,
        "computed_subtotal": round(subtotal, 2),
        "extracted_total": total,
        "computed_total": round(computed_total, 2),
        "gst": gst,
        "discount": discount
    }

def extract_line_items_from_ocr(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract line items from OCR words using heuristics:
    - Look for rows with numbers and amounts
    - Try to group columns: description, qty, unit price, total
    """
    line_items = []
    for i, word in enumerate(words):
        # Try to find a row start (e.g., serial number or HSN/SAC code)
        if word["text"].isdigit():
            # Try to extract columns from nearby words
            row = {"description": "", "qty": None, "unit_price": None, "row_total": None}
            context = words[i:i+8]
            numbers = [w for w in context if any(c.isdigit() for c in w["text"])]
            if len(numbers) >= 3:
                try:
                    row["description"] = " ".join([w["text"] for w in context[1:3]])
                    row["qty"] = float(numbers[1]["text"].replace(",", ""))
                    row["unit_price"] = float(numbers[2]["text"].replace(",", ""))
                    row["row_total"] = float(numbers[3]["text"].replace(",", ""))
                    # Validate: unit_price * qty ≈ row_total
                    valid = abs(row["unit_price"] * row["qty"] - row["row_total"]) < 2.0
                    row["valid"] = valid
                    line_items.append(row)
                except Exception:
                    continue
    return line_items

def process_verification(prefix: str):
    # Load extracted fields and line items
    fields_path = FIELDS_OUTPUT_DIR / f"{prefix}_fields.json"
    lineitems_path = LINEITEMS_OUTPUT_DIR / f"{prefix}_lineitems.json"
    if not fields_path.exists() or not lineitems_path.exists():
        print(f"Required files not found for prefix '{prefix}'.")
        return
    fields = load_json(fields_path)
    line_items = load_json(lineitems_path)
    result = verify_calculations(fields, line_items)
    ensure_dir(VERIFICATION_OUTPUT_DIR)
    out_path = VERIFICATION_OUTPUT_DIR / f"{prefix}_verifiability_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Verification report saved to {out_path.resolve()}")
    return result

def process_line_items(prefix: str):
    ocr_files = sorted(OCR_OUTPUT_DIR.glob(f"{prefix}_page_*_raw.json"))
    if not ocr_files:
        print(f"No OCR outputs found for prefix '{prefix}'.")
        return
    words = []
    for ocr_file in ocr_files:
        words.extend(load_ocr_json(ocr_file))
    line_items = extract_line_items_from_ocr(words)
    ensure_dir(LINEITEMS_OUTPUT_DIR)
    out_path = LINEITEMS_OUTPUT_DIR / f"{prefix}_lineitems.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(line_items, f, indent=2)
    print(f"Extracted line items saved to {out_path.resolve()}")
    return line_items

def get_prefix_from_filename(filename: str) -> str:
    parts = filename.split("_")
    # Remove last 3 parts: page, page number, raw.json
    return "_".join(parts[:-3])

def main():
    all_ocr = list(OCR_OUTPUT_DIR.glob("*_raw.json"))
    prefixes = sorted(set(get_prefix_from_filename(f.name) for f in all_ocr))
    if not prefixes:
        print(f"No OCR outputs found in {OCR_OUTPUT_DIR.resolve()}")
        return
    for prefix in prefixes:
        process_line_items(prefix)

if __name__ == "__main__":
    main()