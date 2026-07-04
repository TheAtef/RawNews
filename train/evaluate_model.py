import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from train_classifier import MultiHeadAraBERT, MODEL_NAME
from prepare_data import load_file_to_df, prepare_multitask_dataset, PROPAGANDA_MAP

def evaluate_propaganda():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("./train/models/fine_tuned_arabert_multitask")
    model = MultiHeadAraBERT.from_pretrained("./train/models/fine_tuned_arabert_multitask")
    model.eval()
    model.cuda()

    print("Loading test data...")
    test_df = load_file_to_df("./clean_data/relabeled_test.jsonl")
    
    true_labels = []
    pred_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            text = str(row.get("optimized_text", ""))
            
            true_label_str = row.get("propaganda_label")
            if true_label_str not in PROPAGANDA_MAP:
                continue
            true_labels.append(PROPAGANDA_MAP[true_label_str])
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=400).to("cuda")
            logits = model(**inputs).logits
            

            pr_logits = logits[0, 3:7] 
            pred_class = torch.argmax(pr_logits).item()
            pred_labels.append(pred_class)

    target_names = list(PROPAGANDA_MAP.keys())
    print("\n--- PROPAGANDA CLASSIFICATION REPORT ---")
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual Human Label')
    plt.xlabel('AI Predicted Label')
    plt.title('Propaganda Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate_propaganda()