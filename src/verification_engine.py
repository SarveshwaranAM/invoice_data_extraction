import json
from pathlib import Path
from typing import List, Dict, Any

FIELDS_OUTPUT_DIR = Path("output/")
LINEITEMS_OUTPUT_DIR = Path("output/")
VERIFICATION_OUTPUT_DIR = Path("output/")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
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

def main():
    # For standalone testing: process all available prefixes
    all_fields = list(FIELDS_OUTPUT_DIR.glob("*_fields.json"))
    prefixes = sorted(set(f.name.replace("_fields.json", "") for f in all_fields))
    if not prefixes:
        print(f"No extracted fields found in {FIELDS_OUTPUT_DIR.resolve()}")
        return
    for prefix in prefixes:
        process_verification(prefix)

if __name__ == "__main__":
    main()