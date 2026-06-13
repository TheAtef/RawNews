# train/prepare_data.py
from __future__ import annotations
import os
import pandas as pd
from datasets import Dataset
from arabert.preprocess import ArabertPreprocessor

PROPAGANDA_MAP = {"neutral": 0, "loaded_language": 1, "propaganda": 2}
STATEMENT_MAP = {"fact": 0, "opinion": 1, "speculation": 2}
ATTRIBUTION_MAP = {"supported_claim": 0, "unsupported_claim": 1}

def load_file_to_df(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target dataset file was not found at: {file_path}")
        
    if file_path.endswith(".jsonl"):
        return pd.read_json(file_path, lines=True)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
def prepare_single_dataset(df: pd.DataFrame, target_task: str) -> Dataset:
    preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")
    
    processed_records = []
    for _, row in df.iterrows():
        title = row.get("title") or ""
        content = row.get("text") or ""
        
        combined_text = f"{title} [SEP] {content}"
        cleaned_text = preprocessor.preprocess(combined_text)
        
        if target_task == "propaganda":
            raw_label = row.get("propaganda_label", "neutral")
            label = PROPAGANDA_MAP.get(raw_label, 0)
        elif target_task == "statement_type":
            raw_label = row.get("statement_type", "fact")
            label = STATEMENT_MAP.get(raw_label, 0)
        elif target_task == "attribution":
            raw_label = row.get("attribution_label", "unsupported_claim")
            label = ATTRIBUTION_MAP.get(raw_label, 1)
        else:
            continue
            
        processed_records.append({
            "text": cleaned_text,
            "label": label
        })
        
    return Dataset.from_list(processed_records)