import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
INPUT_FILE = "staged_prop_attr_stmnt.jsonl"
OUTPUT_DIR = "./train/clean_data"
MAX_WORDS = 350 
MIN_WORDS = 30  
def is_corrupted(text: str) -> bool:
    if not isinstance(text, str) or len(text) == 0:
        return True
    corruption_chars = text.count('ط') + text.count('ظ')
    if (corruption_chars / len(text)) > 0.10:
        return True
    return False
def clean_and_truncate(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        
    return text

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    initial_len = len(df)
    df = df.drop_duplicates()
    
    df = df.dropna(subset=['propaganda_label', 'attribution_label', 'statement_type'])

    df = df[df['statement_type'] != 'uncertain']

    mask_not_corrupted = ~(df['title'].fillna("").apply(is_corrupted) | df['text'].fillna("").apply(is_corrupted))
    df = df[mask_not_corrupted]

    df['text_word_count'] = df['text'].fillna("").apply(lambda x: len(x.split()))
    df = df[df['text_word_count'] >= MIN_WORDS]
    
    dropped_bad_data = initial_len - len(df)
    print(f"   -> Dropped {dropped_bad_data} rows (Duplicates, 'uncertain' class, corrupted text, or too short).")

    print("2. Truncating and preparing optimized_text...")
    df['clean_title'] = df['title'].fillna("").apply(clean_and_truncate)
    df['clean_text'] = df['text'].fillna("").apply(clean_and_truncate)
    df['optimized_text'] = df['clean_title'] + " [SEP] " + df['clean_text']

    cols_to_keep = ['optimized_text', 'propaganda_label', 'attribution_label', 'statement_type']
    df_clean = df[cols_to_keep]

    print("3. Splitting into 85% Train / 15% Test (Stratified)...")
    train_df, test_df = train_test_split(
        df_clean, 
        test_size=0.15, 
        random_state=42, 
        stratify=df_clean['statement_type'] 
    )

    print(f"\nFinal Train Size: {train_df.shape[0]} rows")
    print(f"Final Test Size:  {test_df.shape[0]} rows")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "clean_train.jsonl")
    test_out = os.path.join(OUTPUT_DIR, "clean_test.jsonl")
    
    train_df.to_json(train_out, orient='records', lines=True, force_ascii=False)
    test_df.to_json(test_out, orient='records', lines=True, force_ascii=False)
    print(f"\nData successfully saved to {OUTPUT_DIR}")
    print("You can now safely run: python train_multitask.py")

if __name__ == "__main__":
    main()