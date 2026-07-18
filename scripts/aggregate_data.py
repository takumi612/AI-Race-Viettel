import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATA_DIR
from src.evaluation.trusted_split import development_ids

def aggregate_data(
    input_dir: str | Path = DATA_DIR / "dev" / "input",
    gt_dir: str | Path = DATA_DIR / "dev" / "gt",
    out_txt: str | Path = DATA_DIR / "dev" / "combined_input.txt",
    out_json: str | Path = DATA_DIR / "dev" / "combined_gt.json",
):
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)
    out_txt = Path(out_txt)
    out_json = Path(out_json)
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if not gt_dir.is_dir():
        raise ValueError(f"ground-truth directory does not exist: {gt_dir}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    combined_gt = {}
    
    with out_txt.open("w", encoding="utf-8") as ft:
        for i in development_ids():
            # Xử lý input text
            txt_path = input_dir / f"{i}.txt"
            if txt_path.is_file():
                with txt_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                ft.write(f"================ File {i}.txt ================\n")
                ft.write(content.strip() + "\n\n")
            
            # Xử lý GT json
            json_path = gt_dir / f"{i}.json"
            if json_path.is_file():
                with json_path.open("r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        combined_gt[str(i)] = data
                    except Exception as e:
                        print(f"Error parsing JSON file {i}: {e}")
    
    with out_json.open("w", encoding="utf-8") as fj:
        json.dump(combined_gt, fj, ensure_ascii=False, indent=4)
        
    print(f"Đã tổng hợp toàn bộ input text vào: {out_txt}")
    print(f"Đã tổng hợp toàn bộ GT json vào: {out_json}")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Aggregate supplied development data")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "dev" / "input")
    parser.add_argument("--gt-dir", type=Path, default=DATA_DIR / "dev" / "gt")
    parser.add_argument(
        "--out-txt", type=Path, default=DATA_DIR / "dev" / "combined_input.txt"
    )
    parser.add_argument(
        "--out-json", type=Path, default=DATA_DIR / "dev" / "combined_gt.json"
    )
    args = parser.parse_args()
    aggregate_data(args.input_dir, args.gt_dir, args.out_txt, args.out_json)
