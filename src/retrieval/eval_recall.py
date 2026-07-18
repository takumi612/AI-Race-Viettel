import sys
import json
import logging
from pathlib import Path

# Thêm project root vào sys.path để hỗ trợ import chéo
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.normalizer import TextNormalizer
from src.config import DATA_DIR
from src.evaluation.trusted_split import development_ids

logger = logging.getLogger("EvalRecall")


def select_trusted_ground_truth_files(gt_dir: Path, limit: int) -> list[Path]:
    if limit < 1:
        raise ValueError("limit must be positive")
    trusted_ids = set(development_ids())
    files = [
        path
        for path in gt_dir.glob("*.json")
        if path.stem.isdigit() and int(path.stem) in trusted_ids
    ]
    return sorted(files, key=lambda path: int(path.stem))[:limit]


def eval_recall_on_dataset(gt_dir: str | Path = DATA_DIR / "dev" / "gt", limit=30):
    gt_dir = Path(gt_dir)
    if not gt_dir.is_dir():
        raise ValueError(f"ground-truth directory does not exist: {gt_dir}")
    normalizer = TextNormalizer()
    
    # Khởi tạo retriever
    icd_retriever = HybridRetriever(table_name="icd10")
    rxnorm_retriever = HybridRetriever(table_name="rxnorm")
    
    total_icd_concepts = 0
    hit_icd_concepts = 0
    
    total_rxnorm_concepts = 0
    hit_rxnorm_concepts = 0
    
    # Lấy danh sách file và sắp xếp
    files_to_eval = select_trusted_ground_truth_files(gt_dir, limit)
    
    logger.info(f"Bắt đầu đánh giá Recall@5 trên {len(files_to_eval)} file nhãn chuẩn đầu tiên...")
    
    for file_path in files_to_eval:
        with file_path.open("r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        for ent in gt_data:
            ent_type = ent.get("type")
            ent_text = ent.get("text")
            gold_codes = [c.strip().upper() for c in ent.get("candidates", []) if c.strip()]
            
            if not gold_codes:
                continue  # Bỏ qua các thực thể không có nhãn chuẩn candidates
                
            if ent_type == "CHẨN_ĐOÁN":
                clean_text = normalizer.clean_text(ent_text)
                expanded_text = normalizer.expand_abbreviation(clean_text)
                
                # Retrieve candidates
                preds = icd_retriever.retrieve(expanded_text, top_k=5)
                preds_upper = [p.strip().upper() for p in preds]
                
                total_icd_concepts += 1
                # Kiểm tra xem có mã gold nào nằm trong top 5 dự đoán không
                if any(gc in preds_upper for gc in gold_codes):
                    hit_icd_concepts += 1
                    
            elif ent_type == "THUỐC":
                clean_text = normalizer.clean_text(ent_text)
                clean_drug = normalizer.remove_dosage(clean_text)
                expanded_text = normalizer.expand_abbreviation(clean_drug)
                
                # Retrieve candidates
                preds = rxnorm_retriever.retrieve(expanded_text, top_k=5)
                preds_upper = [p.strip().upper() for p in preds]
                
                total_rxnorm_concepts += 1
                if any(gc in preds_upper for gc in gold_codes):
                    hit_rxnorm_concepts += 1

    # Tính toán kết quả
    recall_icd = hit_icd_concepts / total_icd_concepts if total_icd_concepts > 0 else 0.0
    recall_rxnorm = hit_rxnorm_concepts / total_rxnorm_concepts if total_rxnorm_concepts > 0 else 0.0
    
    total_concepts = total_icd_concepts + total_rxnorm_concepts
    total_hits = hit_icd_concepts + hit_rxnorm_concepts
    overall_recall = total_hits / total_concepts if total_concepts > 0 else 0.0
    
    print("\n============================================================")
    print(f"  EVALUATION RESULTS: RECALL@5 FOR HYBRID RETRIEVER ({icd_retriever.embedding_model_type})")
    print("============================================================")
    print(f"1. DIAGNOSIS (ICD-10): Recall@5 = {recall_icd:.4f} ({hit_icd_concepts}/{total_icd_concepts})")
    print(f"2. DRUG (RxNorm)     : Recall@5 = {recall_rxnorm:.4f} ({hit_rxnorm_concepts}/{total_rxnorm_concepts})")
    print("------------------------------------------------------------")
    print(f"  OVERALL RECALL@5   = {overall_recall:.4f} ({total_hits}/{total_concepts})")
    print("============================================================")
    
    return overall_recall

if __name__ == "__main__":
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DATA_DIR / "dev" / "gt",
        help="Directory containing supplied ground-truth JSON files",
    )
    parser.add_argument("--limit", type=int, default=30, help="Số lượng file đánh giá")
    args = parser.parse_args()
    
    eval_recall_on_dataset(gt_dir=args.gt_dir, limit=args.limit)
