import json 
import os
ORIGINAL_DATASET="train/balanced_clean_fr.jsonl"
FEEDBACK_DATASET="train/approved_feedback_dataset.jsonl"
OUTPUT_DATASET="train/merged_dataset.jsonl"
def load_jsonl(path):
    data=[]
    if not os.path.exists(path):
        return data
    with open(path,"r",encoding="utf-8")as f:
        for line in f:
            line=line.strip()
            if line:
                data.append(json.loads(line))
    return data
def mainKey(record):
    return (
        record.get("title","").strip(),
        record.get("text","").strip()

    )
def merge():
    original=load_jsonl(ORIGINAL_DATASET)
    feedback=load_jsonl(FEEDBACK_DATASET)
    merged={
    }
    for row in original:
        merged[mainKey(row)]=row
    for row in feedback:
        merged[mainKey(row)]=row
    with open(OUTPUT_DATASET,"w",encoding="utf-8")as f:
        for row in merged.values():
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("=" * 50)
    print(f"Original Dataset : {len(original)}")
    print(f"Feedback Dataset : {len(feedback)}")
    print(f"Merged Dataset   : {len(merged)}")
    print("=" * 50)


if __name__ == "__main__":
    merge()
