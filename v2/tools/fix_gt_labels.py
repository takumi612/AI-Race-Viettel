import json
from pathlib import Path
import sys

def main():
    gt_dir = Path(sys.argv[1])
    if not gt_dir.is_dir():
        print(f"Directory not found: {gt_dir}")
        sys.exit(1)

    mapping = {
        "CHẨN_ĐOÁN": "DISEASE",
        "THUỐC": "DRUG",
        "TRIỆU_CHỨNG": "SYMPTOM",
        "KẾT_QUẢ_XÉT_NGHIỆM": "LAB_RESULT",
        "TÊN_XÉT_NGHIỆM": "LAB_NAME"
    }

    modified_count = 0
    total_count = 0

    for json_file in gt_dir.glob("*.json"):
        total_count += 1
        content = json_file.read_text(encoding="utf-8")
        data = json.loads(content)
        modified = False
        
        for item in data:
            if "type" in item and item["type"] in mapping:
                item["type"] = mapping[item["type"]]
                modified = True
                
        if modified:
            json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            modified_count += 1

    print(f"Processed {total_count} files. Modified labels in {modified_count} files.")

if __name__ == "__main__":
    main()
