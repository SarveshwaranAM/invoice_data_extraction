import re
import json
from pathlib import Path
from typing import Dict, Any, List
import spacy

OCR_OUTPUT_DIR = Path("output/ocr/")
FIELDS_OUTPUT_DIR = Path("output/")
SPACY_MODEL = "en_core_web_sm"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_ocr_json(page_json_path: Path) -> List[Dict[str, Any]]:
    with open(page_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_field(patterns: List[str], text: str, field_name: str) -> Dict[str, Any]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            confidence = 0.95  # Heuristic confidence for regex match
            return {"value": value, "confidence": confidence, "present": True}
    return {"value": None, "confidence": 0.0, "present": False}

def extract_fields_from_text(text: str) -> Dict[str, Any]:
    # Patterns for fields
    patterns = {
        "invoice_number": [r"Invoice\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)"],
        "date": [r"Date\s*[:\-]?\s*([0-9]{2,4}[-/][0-9]{1,2}[-/][0-9]{1,4})"],
        "gst_number": [r"GSTIN\s*[:\-]?\s*([0-9A-Z]{15})"],
        "po_number": [r"PO\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)"]
    }
    results = {}
    for field, pats in patterns.items():
        results[field] = extract_field(pats, text, field)
    return results

def extract_bill_ship_to(text: str, nlp) -> Dict[str, Any]:
    # Use spaCy NER to extract organizations and locations
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON")]
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    # Heuristic: first org/location for bill_to, second for ship_to
    bill_to = orgs[0] if orgs else None
    ship_to = orgs[1] if len(orgs) > 1 else None
    bill_addr = locs[0] if locs else None
    ship_addr = locs[1] if len(locs) > 1 else None
    return {
        "bill_to": {"value": bill_to, "confidence": 0.8 if bill_to else 0.0, "present": bool(bill_to)},
        "ship_to": {"value": ship_to, "confidence": 0.8 if ship_to else 0.0, "present": bool(ship_to)},
        "bill_to_address": {"value": bill_addr, "confidence": 0.7 if bill_addr else 0.0, "present": bool(bill_addr)},
        "ship_to_address": {"value": ship_addr, "confidence": 0.7 if ship_addr else 0.0, "present": bool(ship_addr)},
    }

def process_ocr_pages(prefix: str):
    nlp = spacy.load(SPACY_MODEL)
    ocr_files = sorted(OCR_OUTPUT_DIR.glob(f"{prefix}_page_*_raw.json"))
    all_text = ""
    for ocr_file in ocr_files:
        words = load_ocr_json(ocr_file)
        page_text = " ".join([w["text"] for w in words])
        all_text += page_text + "\n"
    fields = extract_fields_from_text(all_text)
    bill_ship = extract_bill_ship_to(all_text, nlp)
    fields.update(bill_ship)
    ensure_dir(FIELDS_OUTPUT_DIR)
    out_path = FIELDS_OUTPUT_DIR / f"{prefix}_fields.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)
    print(f"Extracted fields saved to {out_path.resolve()}")
    return fields

def main():
    # For standalone testing: process all OCR outputs for all prefixes
    all_ocr = list(OCR_OUTPUT_DIR.glob("*_raw.json"))
    prefixes = sorted(set("_".join(f.name.split("_")[:-3]) for f in all_ocr))
    if not prefixes:
        print(f"No OCR outputs found in {OCR_OUTPUT_DIR.resolve()}")
        return
    for prefix in prefixes:
        process_ocr_pages(prefix)

if __name__ == "__main__":
    main()