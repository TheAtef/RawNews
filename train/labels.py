import json
from collections import Counter
file_path = "balanced_clean_fr.jsonl"
def print_propaganda_labels(file):
    labels = []
    skipped_lines = 0
    total_records = 0
    
    with open(file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_records += 1
                try:
                    record = json.loads(line)
                    label = record.get("propaganda_label", "Missing Label")
                    labels.append(label)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    continue
                    
    if not labels:
            print("No labels found or the file is empty.")
            return
    label_counts = Counter(labels)
    total_valid = len(labels)

    print(f"DATASET ANALYSIS: {file}")
    print(f"Total lines processed : {total_records}")
    print(f"Skipped invalid lines : {skipped_lines}")
    print(f"Total valid records   : {total_valid}")
    print(f"{'Propaganda Label':<25} | {'Count':<8} | {'Percentage':<10}")        
    for label, count in label_counts.most_common():
            percentage = (count / total_valid) * 100
            print(f"{str(label):<25} | {count:<8} | {percentage:.2f}%")


if __name__ == "__main__":
    print_propaganda_labels(file_path)