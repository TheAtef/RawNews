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
    EarlyStoppingCallback
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

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

class MultiHeadAraBERT(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        self.dropout = nn.Dropout(0.15)
        
        self.statement_head = nn.Linear(self.config.hidden_size, len(STATEMENT_MAP))
        self.propaganda_head = nn.Linear(self.config.hidden_size+1, len(PROPAGANDA_MAP))
        self.attribution_head = nn.Linear(self.config.hidden_size, len(ATTRIBUTION_MAP))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,loaded_words_ratio=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        pooled_output = self.dropout(pooled_output)
        
        st_logits = self.statement_head(pooled_output)
        # pr_logits = self.propaganda_head(pooled_output)
        at_logits = self.attribution_head(pooled_output)
        if loaded_words_ratio is None:
            loaded_words_ratio = torch.zeros(
                pooled_output.size(0),
                1,
                device=pooled_output.device
            )
        else:
            loaded_words_ratio = loaded_words_ratio.float().view(-1, 1)

        propaganda_features = torch.cat(
            [pooled_output, loaded_words_ratio],
            dim=-1
        )

        pr_logits = self.propaganda_head(propaganda_features)
        
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
        
        task_importance = {
            "statement_type": 1.0,
            "propaganda": 2.0,    
            "attribution": 1.0
        }
        
        for idx, task in enumerate(MULTITASK_ORDER):
            num_labels = len(TASK_LABEL_MAPS[task])
            task_logits = logits[:, start : start + num_labels]
            task_labels = labels[:, idx]
            
            if task in self.class_weights:
                weight = self.class_weights[task].to(device=device, dtype=task_logits.dtype)
                task_loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
            else:
                task_loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                
            task_loss = task_loss_fct(task_logits, task_labels)
            
            weighted_loss = task_loss * task_importance[task]
            losses.append(weighted_loss) 
            start += num_labels

        loss = sum(losses)
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
        weights = np.clip(weights, a_min=0.5, a_max=3.5)
        
        class_weights[task] = torch.tensor(weights, dtype=torch.float)
    return class_weights


def run_classifier_training(
    task_name: str,
    train_path: str = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_train.jsonl")),
    test_path: str = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_test.jsonl")),
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    ratios = train_ds["loaded_words_ratio"]
    neutral_ratios = [
        row["loaded_words_ratio"]
        for row in train_ds
        if row["propaganda_label"] == 0
    ]

    propaganda_ratios = [
        row["loaded_words_ratio"]
        for row in train_ds
        if row["propaganda_label"] == 1
    ]

    print("=" * 50)

    print("Neutral records:", len(neutral_ratios))
    print(
        "Neutral average loaded ratio:",
        sum(neutral_ratios) / len(neutral_ratios)
    )

    print("Propaganda records:", len(propaganda_ratios))
    print(
        "Propaganda average loaded ratio:",
        sum(propaganda_ratios) / len(propaganda_ratios)
    )

    print("=" * 50)

    non_zero_ratios = [
        ratio for ratio in ratios
        if ratio > 0
    ]

    print("Total records:", len(ratios))
    print("Non-zero loaded ratios:", len(non_zero_ratios))

    if non_zero_ratios:
        print("Max loaded ratio:", max(non_zero_ratios))
        print(
            "Average non-zero ratio:",
            sum(non_zero_ratios) / len(non_zero_ratios)
        )
    print(train_ds.column_names)

    print(train_ds[0])

    print(
        "Loaded words ratio:",
        train_ds[0]["loaded_words_ratio"]
    )

    def tokenize_fn(examples):
        return tokenizer(
            text=examples["title"],
            text_pair=examples["content"],
            truncation="only_second",
            max_length=512,
            padding="max_length"
        )

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
        remove_columns=["title", "content", "statement_type_label", "propaganda_label", "attribution_label"],
    )
    tokenized_eval = tokenized_eval.map(
        combine_labels,
        batched=True,
        remove_columns=["title", "content", "statement_type_label", "propaganda_label", "attribution_label"],
    )

    model = MultiHeadAraBERT(MODEL_NAME)

    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1") 

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.asarray(labels)
        
        st_len = len(STATEMENT_MAP)
        pr_len = len(PROPAGANDA_MAP)
        
        statement_preds = np.argmax(logits[:, :st_len], axis=-1)
        propaganda_preds = np.argmax(logits[:, st_len : st_len + pr_len], axis=-1)
        attribution_preds = np.argmax(logits[:, st_len + pr_len :], axis=-1)

        st_acc = metric.compute(predictions=statement_preds, references=labels[:, 0])["accuracy"]
        pr_acc = metric.compute(predictions=propaganda_preds, references=labels[:, 1])["accuracy"]
        at_acc = metric.compute(predictions=attribution_preds, references=labels[:, 2])["accuracy"]
        
        pr_f1 = f1_metric.compute(predictions=propaganda_preds, references=labels[:, 1], average="macro")["f1"]

        avg_accuracy = (st_acc + pr_acc + at_acc) / 3.0

        return {
            "statement_acc": st_acc,
            "propaganda_acc": pr_acc,
            "prop_f1_macro": pr_f1, 
            "attribution_acc": at_acc,
            "avg_accuracy": avg_accuracy,
        }

    training_args = TrainingArguments(
        output_dir=f"./train/checkpoints/{task_name}_model",
        learning_rate=1.5e-5,         
        num_train_epochs=10,              
        per_device_train_batch_size=8,    
        per_device_eval_batch_size=8,      
        gradient_accumulation_steps=4,     
        max_grad_norm=1.0,                
        weight_decay=0.05,
        warmup_ratio=0.10,              
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_avg_accuracy", 
        greater_is_better=True,
        fp16=True,                     
        logging_steps=50,
        dataloader_num_workers=4, 
        remove_unused_columns=False         
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )

    print(f"Starting training session on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    trainer.train()

    save_path = f"./train/models/fine_tuned_arabert_{task_name}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model successfully saved to: {save_path}")


if __name__ == "__main__":
    run_classifier_training(task_name="multitask")