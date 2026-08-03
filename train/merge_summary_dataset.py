from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset



FEEDBACK_DATASET = Path("train/approved_summary_feedback_dataset.jsonl")
OUTPUT_DATASET = Path("train/merged_summary_dataset.jsonl")
def load_jsonl(path: Path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def merge_summary_dataset():
    print("merging summary datset has started")
    dataset = load_dataset("arbml/AraSum",split="train")

    merged={}
    print("Loading original dataset...")

    added_feedback=0

    for row in dataset:
        article = (row.get("article") or "").strip()
        summary = (row.get("summary") or "").strip()
        key=(article,summary)
        merged[key]={
            "article":article,
            "summary":summary
        }
    print(f"Original dataset: {len(merged)} samples")
    added = 0
    skipped = 0

    print("Loading approved feedback...")

    for row in load_jsonl(FEEDBACK_DATASET):

        article = (row.get("article") or "").strip()
        summary = (row.get("summary") or "").strip()

        key = (article, summary)

        if key not in merged:

            merged[key] = {
                "article": article,
                "summary": summary
            }

            added += 1

        else:
            skipped += 1
    print(f"Added feedback samples : {added}")
    print(f"Skipped duplicates     : {skipped}")

    print("Saving merged dataset...")

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:

        for row in merged.values():
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    print(f"Added feedback samples : {added_feedback}")
    print(f"Final dataset size     : {len(merged)}")
    print(f"Saved to               : {OUTPUT_DATASET}")


if __name__ == "__main__":
    merge_summary_dataset()
