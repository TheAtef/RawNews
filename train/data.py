import json
from transformers import pipeline

def re_evaluate_dataset(input_filepath, output_filepath):
    print("Loading NLP model...")
    classifier = pipeline(
        "zero-shot-classification", 
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    candidate_labels = [
        "neutral", "propaganda", "loaded_language", 
        "sensationalism", "false_dichotomy", "fear_appeal", 
        "doubt_casting", "exaggeration", "stereotyping"
    ]
    processed_count = 0
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            result = classifier(text, candidate_labels=candidate_labels)
            
            predicted_label = result['labels'][0]
            confidence_score = result['scores'][0]
            
            if confidence_score > 0.4:
                record["propaganda_label"] = predicted_label
            else:
                record["propaganda_label"] = "neutral" if "neutral" in result['labels'][:2] else record.get("propaganda_label")
            
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} records...")
    print(f"Re-labeling complete. Saved to '{output_filepath}'.")

if __name__ == "__main__":
    re_evaluate_dataset("balanced_clean_fr_fixed.jsonl", "balanced_clean_fr_re_labeled.jsonl")