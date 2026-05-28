# test_final_dataset.py
import json
import os
import random
from typing import Dict, List, Set

EXPECTED_PROPAGANDA = {
    "neutral", 
    "false_dichotomy", 
    "fear_appeal", 
    "loaded_language", 
    "stereotyping", 
    "doubt_casting",   
    "exaggeration",     
    "sensationalism"   
}

EXPECTED_ATTRIBUTION = {
    "unsupported_claim", 
    "quote_present", 
    "direct_source", 
    "official_statement", 
    "anonymous_source"
}
def run_dataset_integrity_test(dataset_dir: str):

    splits = {
        "Train (Balanced)": "balanced_train_sentences.jsonl",
        "Validation (Raw)": "clean_val_sentences.jsonl",
        "Test (Raw)": "clean_test_sentences.jsonl"
    }

    print("="*80)
    print("RUNNING FINAL DATASET VERIFICATION & INTEGRITY SUITE")
    print("="*80)

    all_passed = True
    samples_for_review: List[dict] = []

    for split_name, filename in splits.items():
        file_path = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: {split_name} file not found at: {file_path}")
            all_passed = False
            continue

        total_records = 0
        null_fields_detected = 0
        invalid_types_detected = 0
        invalid_labels_detected = 0
        
        encountered_propaganda: Set[str] = set()
        encountered_attribution: Set[str] = set()
        encountered_statements: Set[str] = set()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    total_records += 1
                    
                    if split_name == "Train (Balanced)" and len(samples_for_review) < 3:
                        if record.get("propaganda_label") != "neutral":
                            samples_for_review.append(record)
                    required_keys = ["text", "source", "source_bias", "region", "propaganda_label", 
                                     "attribution_label", "statement_type", "verified", "reliability_score", 
                                     "title", "date"]
                    
                    for k in required_keys:
                        if k not in record or record[k] is None:
                            null_fields_detected += 1
                    
                    if not isinstance(record.get("text", ""), str) or len(record.get("text", "")) < 10:
                        invalid_types_detected += 1
                    if not isinstance(record.get("verified", None), bool):
                        invalid_types_detected += 1
                    if not isinstance(record.get("reliability_score", None), (int, float)):
                        invalid_types_detected += 1
                    p_label = record.get("propaganda_label")
                    a_label = record.get("attribution_label")
                    s_label = record.get("statement_type")
                    if p_label not in EXPECTED_PROPAGANDA:
                        invalid_labels_detected += 1
                    if a_label not in EXPECTED_ATTRIBUTION:
                        invalid_labels_detected += 1

                    encountered_propaganda.add(p_label)
                    encountered_attribution.add(a_label)
                    encountered_statements.add(s_label)

                except json.JSONDecodeError:
                    print(f"  [!] Syntax Error: Invalid JSON syntax on line {line_idx} in {filename}")
                    all_passed = False

        print(f"\n[*] Results for split: '{split_name}'")
        print(f"  - Location             : {file_path}")
        print(f"  - Total Sentences      : {total_records}")
        
        if null_fields_detected == 0 and invalid_types_detected == 0 and invalid_labels_detected == 0:
            print("  - Schema Validation    : PASSED (Zero missing fields or invalid structures)")
        else:
            print(f"  - Schema Validation    : FAILED")
            print(f"    * Missing/Null Fields: {null_fields_detected}")
            print(f"    * Invalid Data Types : {invalid_types_detected}")
            print(f"    * Out-of-Bounds Labels: {invalid_labels_detected}")
            all_passed = False

        print(f"  - Unique Prop. Labels  : {sorted(list(encountered_propaganda))}")
        print(f"  - Unique Attr. Labels  : {sorted(list(encountered_attribution))}")
        print(f"  - Unique Statement Kinds: {sorted(list(encountered_statements))}")

    if len(samples_for_review) < 3 and os.path.exists(os.path.join(dataset_dir, splits["Train (Balanced)"])):
        with open(os.path.join(dataset_dir, splits["Train (Balanced)"]), "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record not in samples_for_review:
                    samples_for_review.append(record)
                if len(samples_for_review) >= 3:
                    break

    if samples_for_review:
        print("\n" + "="*80)
        print("MANUAL CONTENT SPOT-CHECK (RANDOM SAMPLE REVIEW)")
        print("="*80)
        for idx, sample in enumerate(samples_for_review, 1):
            print(f"\n[Sample #{idx}]")
            print(f"  Text              : {sample['text']}")
            print(f"  Title             : {sample['title']}")
            print(f"  Source (Bias/Reg) : {sample['source']} ({sample['source_bias']} / {sample['region']})")
            print(f"  Propaganda Label  : {sample['propaganda_label']}")
            print(f"  Attribution Label : {sample['attribution_label']}")
            print(f"  Statement Type    : {sample['statement_type']}")
            print(f"  Reliability Score : {sample['reliability_score']}")
            print(f"  Date              : {sample['date']}")
            print("-" * 50)

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] Your final dataset is fully ready for model training.")
    else:
        print("[!] Attention Required: Some validation steps failed. See reports above.")
    print("="*80)


if __name__ == "__main__":
    run_dataset_integrity_test("./training_ready_dataset_balanced")