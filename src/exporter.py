import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

OUTPUT_DIR = Path("output/")
EXPORT_JSON = OUTPUT_DIR / "extracted_data.json"
EXPORT_XLSX = OUTPUT_DIR / "extracted_data.xlsx"
VERIFICATION_REPORT = OUTPUT_DIR / "verifiability_report.json"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_prefixes() -> List[str]:
    # Find all prefixes based on *_fields.json files
    field_files = list(OUTPUT_DIR.glob("*_fields.json"))
    prefixes = sorted(set(f.name.replace("_fields.json", "") for f in field_files))
    return prefixes

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def export_to_json(all_data: List[Dict[str, Any]]):
    with open(EXPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"Exported all extracted data to {EXPORT_JSON.resolve()}")

def export_to_excel(all_data: List[Dict[str, Any]]):
    # Flatten for Excel: one sheet for fields, one for line items
    fields_rows = []
    line_items_rows = []
    for doc in all_data:
        prefix = doc.get("prefix", "")
        fields = doc.get("fields", {})
        fields_row = {"prefix": prefix}
        fields_row.update({k: v.get("value") if isinstance(v, dict) else v for k, v in fields.items()})
        fields_rows.append(fields_row)
        for item in doc.get("line_items", []):
            item_row = {"prefix": prefix}
            item_row.update(item)
            line_items_rows.append(item_row)
    with pd.ExcelWriter(EXPORT_XLSX) as writer:
        pd.DataFrame(fields_rows).to_excel(writer, sheet_name="Fields", index=False)
        pd.DataFrame(line_items_rows).to_excel(writer, sheet_name="LineItems", index=False)
    print(f"Exported all extracted data to {EXPORT_XLSX.resolve()}")

def export_verification_reports(reports: List[Dict[str, Any]]):
    with open(VERIFICATION_REPORT, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"Exported verifiability report to {VERIFICATION_REPORT.resolve()}")

def main():
    ensure_dir(OUTPUT_DIR)
    prefixes = get_prefixes()
    if not prefixes:
        print("No extracted data found to export.")
        return

    all_data = []
    verification_reports = []
    for prefix in prefixes:
        fields_path = OUTPUT_DIR / f"{prefix}_fields.json"
        lineitems_path = OUTPUT_DIR / f"{prefix}_lineitems.json"
        verification_path = OUTPUT_DIR / f"{prefix}_verifiability_report.json"
        fields = load_json(fields_path) if fields_path.exists() else {}
        line_items = load_json(lineitems_path) if lineitems_path.exists() else []
        verification = load_json(verification_path) if verification_path.exists() else {}
        all_data.append({
            "prefix": prefix,
            "fields": fields,
            "line_items": line_items
        })
        if verification:
            verification_reports.append({"prefix": prefix, **verification})

    export_to_json(all_data)
    export_to_excel(all_data)
    export_verification_reports(verification_reports)

if __name__ == "__main__":
    main()