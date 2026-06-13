from __future__ import annotations
import sys
import os
import torch
import structlog
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.prepare_data import load_file_to_df

logger = structlog.get_logger(__name__)

def format_gemma_prompt(df, text_col: str = "text", summary_col: str = "summary") -> Dataset:
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
            "<start_of_turn>model\n"
            "التقرير الصحفي الموحد والحيادي:\n"
            f"{target_summary}\n"
            "<end_of_turn>"
        )
        formatted_prompts.append({"prompt": prompt})
        
    return Dataset.from_list(formatted_prompts)

def train_gemma_summarizer(
    train_path: str = "./data/balanced_train_sentences.jsonl",
    test_path: str = "./data/clean_test_sentences.jsonl"
):
    model_id = "google/gemma-2-9b-it"
    
    print(f"Loading summarization training set from: {train_path}")
    train_df = load_file_to_df(train_path)
    
    print(f"Loading summarization testing set from: {test_path}")
    test_df = load_file_to_df(test_path)
    
    train_ds = format_gemma_prompt(train_df, text_col="text", summary_col="summary")
    eval_ds = format_gemma_prompt(test_df, text_col="text", summary_col="summary")
    
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Ensure your JSON/JSONL files have valid text content.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
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
        output_dir="./train/checkpoints/gemma2_summarizer",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataset_text_field="prompt",
        max_seq_length=1524
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    print("Beginning Gemma 2 QLoRA Summarizer Training...")
    trainer.train()
    
    save_path = "./train/models/gemma2_arabic_summarizer_adapter"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Summarizer adapters successfully saved to: {save_path}")

if __name__ == "__main__":
    train_gemma_summarizer()