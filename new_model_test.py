import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

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
    re_evaluate_dataset("train/clean_test_sentences_re_labeled.jsonl", "train/clean_test_sentences_re_labeled_fr.jsonl")


# from datasets import load_dataset

# dataset = load_dataset("arbml/AraSum", split="train")

# shuffled_dataset = dataset.shuffle(seed=42)

# total_rows = shuffled_dataset.num_rows
# batch_size = 1000
# num_batches = 15

# step = (total_rows - batch_size) // (num_batches - 1)

# selected_indices = []
# for i in range(num_batches):
#     start_idx = i * step
#     end_idx = start_idx + batch_size
#     selected_indices.extend(range(start_idx, end_idx))

# final_dataset = shuffled_dataset.select(selected_indices)

# print(f"Original size: {len(dataset)}")
# print(f"New size: {len(final_dataset)}")
# print(final_dataset)