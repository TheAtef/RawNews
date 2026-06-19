import json
from core.sources_list import ARABIC_NEWS_SOURCES

data = []
unknown = []
unknown_count = 0
solved_count = 0
with open("train/balanced_train_sentences.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        if item['source_bias'] == "unknown":
            unknown_count += 1
            for source in ARABIC_NEWS_SOURCES:
                if item['source'] == source.name or item['source'] == source.name_ar:
                    print(f"Matched unknown source '{item['source']}' to known source '{source.name}' with bias '{source.political_lean}' and region '{source.region}'.")
                    item['source_bias'] = source.political_lean
                    item['region'] = source.region
                    data.append(item)
                    solved_count += 1
                    break
            if item['source_bias'] == "unknown":
                unknown.append(item)
        else:
            data.append(item)
    
with open("train/balanced_clean_fr.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
with open("unknown_data.jsonl", "w", encoding="utf-8") as f:
    for item in unknown:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Unknown count: {unknown_count}")
print(f"Solved count: {solved_count}")
print(f"Unsolved count: {unknown_count - solved_count}")