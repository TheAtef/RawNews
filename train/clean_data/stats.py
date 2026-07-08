import json
import pandas as pd

input_file = "relabeled_train.jsonl"

def print_dataset_stats(file_path):
    print(f"Loading data from {file_path}...\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        df = pd.DataFrame(data)
        
        total_rows = len(df)
        print("="*50)
        print(f"📊 DATASET OVERVIEW")
        print("="*50)
        print(f"Total Records: {total_rows:,}\n")

        label_columns = [
            "propaganda_label", 
            "statement_type", 
            "attribution_label"
        ]

        for col in label_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in data!")
                continue
                
            print(f" {col.upper()} DISTRIBUTION:")
            print("-" * 30)
            
       
            counts = df[col].value_counts(dropna=False)
            percentages = df[col].value_counts(normalize=True, dropna=False) * 100
            
            stats_df = pd.DataFrame({
                'Count': counts, 
                'Percentage (%)': percentages.round(2)
            })
            
            print(stats_df.to_string())
            print("\n")
            
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}. Make sure the script is finished running!")

print_dataset_stats(input_file)