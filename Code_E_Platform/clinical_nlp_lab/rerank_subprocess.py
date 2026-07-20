from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run Clinical NLP Reranking and Assertion in a subprocess.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input queries JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output results JSON file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="vLLM model name.")
    args = parser.parse_args()

    # 1. Read input queries FIRST (before loading heavy LLM engine)
    logging.info(f"Loading input queries from: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rerank_queries = data.get("rerank_queries", [])
    assertion_queries = data.get("assertion_queries", [])

    # Early exit if no queries
    if not rerank_queries and not assertion_queries:
        logging.info("No queries to process. Writing empty output.")
        output_data = {
            "rerank_results": [],
            "assertion_results": []
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        return

    # 2. Import and initialize LLM engine
    try:
        from clinical_nlp_lab.reranker import ClinicalLLMReranker
        from clinical_nlp_lab.assertions import ClinicalLLMAssertionPredictor

        logging.info(f"Starting LLM engine with model: {args.model_name}")
        reranker = ClinicalLLMReranker(model_name=args.model_name)
        llm_assertion = ClinicalLLMAssertionPredictor(reranker.llm)
    except ImportError as e:
        logging.warning(f"Could not initialize reranker or assertions: {e}. Skipping LLM subprocess.")
        output_data = {
            "rerank_results": [],
            "assertion_results": []
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        return

    rerank_results = []
    assertion_results = []

    # 3. Run Reranker
    if rerank_queries:
        logging.info(f"Running rerank batch for {len(rerank_queries)} queries...")
        rerank_results = reranker.rerank_batch(rerank_queries)
    else:
        logging.info("No rerank queries provided.")

    # 4. Run Assertion
    if assertion_queries:
        logging.info(f"Running assertion batch for {len(assertion_queries)} queries...")
        raw_assertions = llm_assertion.predict_batch(assertion_queries)
        assertion_results = [axes.labels() for axes in raw_assertions]
    else:
        logging.info("No assertion queries provided.")

    # 5. Cleanup LLM engine immediately
    logging.info("Cleaning up LLM engine and freeing VRAM...")
    reranker.destroy()

    # 6. Save outputs
    output_data = {
        "rerank_results": rerank_results,
        "assertion_results": assertion_results
    }
    logging.info(f"Saving output results to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logging.info("Subprocess execution completed successfully.")

if __name__ == "__main__":
    main()
