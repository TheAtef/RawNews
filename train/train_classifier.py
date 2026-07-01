from __future__ import annotations
import collections
import collections.abc

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.prepare_data import (
    ATTRIBUTION_MAP,
    PROPAGANDA_MAP,
    STATEMENT_MAP,
    load_file_to_df,
    prepare_multitask_dataset,
    prepare_single_dataset,
)

TASK_LABEL_MAPS = {
    "statement_type": STATEMENT_MAP,
    "propaganda": PROPAGANDA_MAP,
    "attribution": ATTRIBUTION_MAP,
}

MULTITASK_ORDER = ["statement_type", "propaganda", "attribution"]


def build_label_maps(task_name: str):
    if task_name == "multitask":
        id2label = {}
        start = 0
        for task in MULTITASK_ORDER:
            task_map = TASK_LABEL_MAPS[task]
            for label, idx in sorted(task_map.items(), key=lambda item: item[1]):
                id2label[start + idx] = f"{task}:{label}"
            start += len(task_map)
        return None, id2label

    label2id = TASK_LABEL_MAPS[task_name]
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss()
        losses = []
        start = 0
        for task in MULTITASK_ORDER:
            num_labels = len(TASK_LABEL_MAPS[task])
            task_logits = logits[:, start : start + num_labels]
            task_labels = labels[:, MULTITASK_ORDER.index(task)]
            losses.append(loss_fct(task_logits, task_labels))
            start += num_labels

        loss = sum(losses) / len(losses)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels", None)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        return (None, logits, labels)


def run_classifier_training(
    task_name: str,
    num_labels: int | None = None,
    train_path: str = os.path.join(script_dir, "balanced_clean_fr_re_labeled_prop_and_statement_type.jsonl"),
    test_path: str = os.path.join(script_dir, "clean_test_sentences_re_labeled_fr.jsonl"),
):
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading training data from: {train_path}")
    train_df = load_file_to_df(train_path)
    print(f"Loading testing data from: {test_path}")
    test_df = load_file_to_df(test_path)

    if task_name == "multitask":
        train_ds = prepare_multitask_dataset(train_df)
        eval_ds = prepare_multitask_dataset(test_df)
    else:
        train_ds = prepare_single_dataset(train_df, task_name)
        eval_ds = prepare_single_dataset(test_df, task_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True)

    if task_name == "multitask":
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

        num_labels = sum(len(TASK_LABEL_MAPS[task]) for task in MULTITASK_ORDER)
        _, id2label = build_label_maps(task_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
        )
    else:
        tokenized_train = tokenized_train.remove_columns(["text"])
        tokenized_eval = tokenized_eval.remove_columns(["text"])

        label2id, id2label = build_label_maps(task_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task_name == "multitask":
            labels = np.asarray(labels)
            statement_preds = np.argmax(logits[:, : len(STATEMENT_MAP)], axis=-1)
            propaganda_preds = np.argmax(
                logits[:, len(STATEMENT_MAP) : len(STATEMENT_MAP) + len(PROPAGANDA_MAP)],
                axis=-1,
            )
            attribution_preds = np.argmax(logits[:, -len(ATTRIBUTION_MAP) :], axis=-1)

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

        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=f"./train/checkpoints/{task_name}_model",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_avg_accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer_class = MultiTaskTrainer if task_name == "multitask" else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(f"Starting training session on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    trainer.train()

    save_path = f"./train/models/fine_tuned_arabert_{task_name}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model successfully saved to: {save_path}")

if __name__ == "__main__":
    run_classifier_training(task_name="multitask")
