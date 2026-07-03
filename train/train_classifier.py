from __future__ import annotations
import collections
import collections.abc
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import evaluate
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    BertModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

from train.prepare_data import (
    ATTRIBUTION_MAP,
    PROPAGANDA_MAP,
    STATEMENT_MAP,
    load_file_to_df,
    prepare_multitask_dataset,
)

TASK_LABEL_MAPS = {
    "statement_type": STATEMENT_MAP,
    "propaganda": PROPAGANDA_MAP,
    "attribution": ATTRIBUTION_MAP,
}

MULTITASK_ORDER = ["statement_type", "propaganda", "attribution"]


class MultiHeadAraBERT(nn.Module):
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv02"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        self.statement_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, len(STATEMENT_MAP))
        )
        
        self.propaganda_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, len(PROPAGANDA_MAP))
        )
        
        self.attribution_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, len(ATTRIBUTION_MAP))
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        st_logits = self.statement_head(pooled_output)
        pr_logits = self.propaganda_head(pooled_output)
        at_logits = self.attribution_head(pooled_output)
        
        logits = torch.cat([st_logits, pr_logits, at_logits], dim=-1)
        
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = cls()
        if os.path.isdir(pretrained_model_name_or_path):
            state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location="cpu")
                model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))


class MultiTaskTrainer(Trainer):
    def __init__(self, class_weights: dict[str, torch.Tensor] | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        losses = []
        start = 0
        device = logits.device
        
        task_scales = {
            "statement_type": 1.2,
            "propaganda": 3.0,
            "attribution": 0.5
        }
        
        for idx, task in enumerate(MULTITASK_ORDER):
            num_labels = len(TASK_LABEL_MAPS[task])
            task_logits = logits[:, start : start + num_labels]
            task_labels = labels[:, idx]
            
            if task in self.class_weights:
                weight = self.class_weights[task].to(device=device, dtype=task_logits.dtype)
                task_loss_fct = nn.CrossEntropyLoss(weight=weight)
            else:
                task_loss_fct = nn.CrossEntropyLoss()
                
            task_loss = task_loss_fct(task_logits, task_labels)
            scaled_loss = task_loss * task_scales[task]
            losses.append(scaled_loss)
            start += num_labels

        loss = sum(losses) / len(losses)
        return (loss, outputs) if return_outputs else loss


def calculate_task_class_weights(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    class_weights = {}
    valid_mask = (
        df["statement_type"].isin(STATEMENT_MAP.keys()) &
        df["propaganda_label"].isin(PROPAGANDA_MAP.keys()) &
        df["attribution_label"].isin(ATTRIBUTION_MAP.keys())
    )
    clean_df = df[valid_mask]
    
    for task in MULTITASK_ORDER:
        col_name = "propaganda_label" if task == "propaganda" else "attribution_label" if task == "attribution" else "statement_type"
        labels_list = clean_df[col_name].map(TASK_LABEL_MAPS[task]).values
        classes = np.array(list(TASK_LABEL_MAPS[task].values()))
        
        weights = compute_class_weight("balanced", classes=classes, y=labels_list)
        class_weights[task] = torch.tensor(weights, dtype=torch.float)
    return class_weights


def run_classifier_training(
    task_name: str,
    num_labels: int | None = None,
    train_path: str = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_train.jsonl")),
    test_path: str = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_test.jsonl")),
):
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading training data from: {train_path}")
    train_df = load_file_to_df(train_path)
    print(f"Loading testing data from: {test_path}")
    test_df = load_file_to_df(test_path)

    class_weights = None
    if task_name == "multitask":
        print("Calculating balanced class weights for multi-task training...")
        class_weights = calculate_task_class_weights(train_df)

    train_ds = prepare_multitask_dataset(train_df)
    eval_ds = prepare_multitask_dataset(test_df)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=400)

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True)

    def combine_labels(examples):
        return {
            "labels": [
                [st, pr, at]
                for st, pr, at in zip(
                    examples["statement_type_label"],
                    examples["propaganda_label"],
                    examples["attribution_label"],
                )
            ]
        }

    tokenized_train = tokenized_train.map(
        combine_labels,
        batched=True,
        remove_columns=["text", "statement_type_label", "propaganda_label", "attribution_label"],
    )
    tokenized_eval = tokenized_eval.map(
        combine_labels,
        batched=True,
        remove_columns=["text", "statement_type_label", "propaganda_label", "attribution_label"],
    )

    model = MultiHeadAraBERT(model_name)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.asarray(labels)
        
        st_len = len(STATEMENT_MAP)
        pr_len = len(PROPAGANDA_MAP)
        
        statement_preds = np.argmax(logits[:, :st_len], axis=-1)
        propaganda_preds = np.argmax(logits[:, st_len : st_len + pr_len], axis=-1)
        attribution_preds = np.argmax(logits[:, st_len + pr_len :], axis=-1)

        statement_accuracy = metric.compute(predictions=statement_preds, references=labels[:, 0])["accuracy"]
        propaganda_accuracy = metric.compute(predictions=propaganda_preds, references=labels[:, 1])["accuracy"]
        attribution_accuracy = metric.compute(predictions=attribution_preds, references=labels[:, 2])["accuracy"]
        avg_accuracy = (statement_accuracy + propaganda_accuracy + attribution_accuracy) / 3.0

        return {
            "statement_accuracy": statement_accuracy,
            "propaganda_accuracy": propaganda_accuracy,
            "attribution_accuracy": attribution_accuracy,
            "avg_accuracy": avg_accuracy,
        }

    training_args = TrainingArguments(
        output_dir=f"./train/checkpoints/{task_name}_model",
        learning_rate=2e-5,               
        num_train_epochs=5,                
        per_device_train_batch_size=2,    
        per_device_eval_batch_size=8,      
        gradient_accumulation_steps=16,    
        weight_decay=0.01,
        warmup_ratio=0.1,                
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_propaganda_accuracy", 
        greater_is_better=True,
        fp16=True,                     
        logging_steps=10,
        dataloader_num_workers=4,          
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    print(f"Starting training session on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    trainer.train()

    save_path = f"./train/models/fine_tuned_arabert_{task_name}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model successfully saved to: {save_path}")


if __name__ == "__main__":
    run_classifier_training(task_name="multitask")