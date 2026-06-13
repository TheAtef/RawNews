from __future__ import annotations
import sys
import os
import torch
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.prepare_data import load_file_to_df, prepare_single_dataset

def run_classifier_training(
    task_name: str, 
    num_labels: int,
    train_path: str = "./balanced_train_sentences.jsonl",
    test_path: str = "./clean_test_sentences.jsonl"
):
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading training data from: {train_path}")
    train_df = load_file_to_df(train_path)
    print(f"Loading testing data from: {test_path}")
    test_df = load_file_to_df(test_path)
    
    train_ds = prepare_single_dataset(train_df, task_name)
    eval_ds = prepare_single_dataset(test_df, task_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        
    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
        
    training_args = TrainingArguments(
        output_dir=f"./train/checkpoints/{task_name}_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )
    
    print(f"Starting training session on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    trainer.train()
    
    save_path = f"./train/models/fine_tuned_arabert_{task_name}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model successfully saved to: {save_path}")

if __name__ == "__main__":
    run_classifier_training(task_name="statement_type", num_labels=3)