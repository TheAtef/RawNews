from __future__ import annotations
import collections
import collections.abc

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

# get access token from higgingface to download the gemma models.
# access_token = ''
# from huggingface_hub import login
# login(access_token)

import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import torch
torch.cuda.empty_cache()
import structlog
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.prepare_data import load_file_to_df

logger = structlog.get_logger(__name__)

def format_gemma_prompt(df, text_col: str = "text", summary_col: str = "summary") -> Dataset:
    if isinstance(df, Dataset):
        df = df.to_pandas()

    formatted_prompts = []
    for _, row in df.iterrows():
        article_text = row.get(text_col) or ""
        target_summary = row.get(summary_col) or row.get("title") or ""
        
        if not article_text.strip() or not target_summary.strip():
            continue
            
        prompt = (
            "<start_of_turn>user\n"
            "قم بتلخيص النص الإخباري التالي صياغة حيادية وموضوعية دقيقة باللغة العربية:\n"
            f"\"{article_text}\"\n"
            "<end_of_turn>\n"
        )
        completion = (
            "<start_of_turn>model\n"
            "التقرير الصحفي الموحد والحيادي:\n"
            f"{target_summary}\n"
            "<end_of_turn>"
        )
        formatted_prompts.append({"prompt": prompt, "completion": completion})
        
    return Dataset.from_list(formatted_prompts)

def train_gemma_summarizer():
    # model_id = "google/gemma-4-E2B-it" # 10.2GB
    model_id = "google/gemma-3-1b-it" # 2GB    
    dataset = load_dataset("arbml/AraSum", split="train")

    shuffled_dataset = dataset.shuffle(seed=11)

    total_rows = shuffled_dataset.num_rows
    batch_size = 250
    num_batches = 24

    step = (total_rows - batch_size) // (num_batches - 1)

    selected_indices = []
    for i in range(num_batches):
        start_idx = i * step
        end_idx = start_idx + batch_size
        selected_indices.extend(range(start_idx, end_idx))

    final_dataset = shuffled_dataset.select(selected_indices)

    print(f"Original size: {len(dataset)}")
    print(f"New size: {len(final_dataset)}")
    print(final_dataset)
    
    
    split = final_dataset.train_test_split(test_size=0.1, seed=11)
    train_df = split['train']
    test_df = split['test']
    
    del split
    del final_dataset
    
    train_ds = format_gemma_prompt(train_df, text_col="article", summary_col="summary")
    eval_ds = format_gemma_prompt(test_df, text_col="article", summary_col="summary")
    
    del train_df
    del test_df
    
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Ensure your JSON/JSONL files have valid text content.")

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True
    # )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        device_map="cuda" # "auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    training_args = SFTConfig(
        output_dir="./train/checkpoints/gemma3_summarizer_2",
        save_strategy="no",
        # save_total_limit=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        num_train_epochs=1,
        dataset_text_field="prompt",
        fp16=True,
        # max_length=1024,
        logging_steps=20
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("Beginning Gemma 3 Summarizer Training...")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()
    
    save_path = "./train/models/gemma3_1b_arabic_summarizer_adapter_2"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Summarizer adapters successfully saved to: {save_path}")

if __name__ == "__main__":
    train_gemma_summarizer()