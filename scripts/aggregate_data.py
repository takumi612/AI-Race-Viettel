import os
import json
import sys

def aggregate_data():
    input_dir = r"D:\AI Race Viettel\data\dev\input"
    gt_dir = r"D:\AI Race Viettel\data\dev\gt"
    out_txt = r"D:\AI Race Viettel\data\dev\combined_input.txt"
    out_json = r"D:\AI Race Viettel\data\dev\combined_gt.json"
    
    combined_gt = {}
    
    with open(out_txt, "w", encoding="utf-8") as ft:
        for i in range(1, 201):
            # Xử lý input text
            txt_path = os.path.join(input_dir, f"{i}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                ft.write(f"================ File {i}.txt ================\n")
                ft.write(content.strip() + "\n\n")
            
            # Xử lý GT json
            json_path = os.path.join(gt_dir, f"{i}.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        combined_gt[str(i)] = data
                    except Exception as e:
                        print(f"Error parsing JSON file {i}: {e}")
    
    with open(out_json, "w", encoding="utf-8") as fj:
        json.dump(combined_gt, fj, ensure_ascii=False, indent=4)
        
    print(f"Đã tổng hợp toàn bộ input text vào: {out_txt}")
    print(f"Đã tổng hợp toàn bộ GT json vào: {out_json}")

if __name__ == "__main__":
    aggregate_data()
