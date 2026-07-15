from __future__ import annotations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pyexpat import model
import sys
import gc
import torch
import evaluate
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datetime import datetime
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

from train.prepare_data import load_file_to_df, prepare_multitask_dataset
from train.prompt_utils import build_messages, parse_output

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 512
MAX_EVAL_INPUT_TOKENS = 512

DEFAULT_TRAIN_PATH = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_train.jsonl"))
DEFAULT_TEST_PATH = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_test.jsonl"))


def format_prompts(batch, tokenizer):
    formatted_texts = []
    for title, content, loaded_ratio, st, pr, at in zip(
        batch["title"],
        batch["content"],
        batch["loaded_words_ratio"],
        batch["statement_type_label"],
        batch["propaganda_label"],
        batch["attribution_label"]
    ):
        messages = build_messages(title, content, loaded_ratio, st, pr, at, include_answer=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append(text)

    return {"text": formatted_texts}


def run_classifier_training(task_name: str = "multitask", train_path=None, test_path=None):
    if train_path is None:
        train_path = DEFAULT_TRAIN_PATH

    if test_path is None:
        test_path = DEFAULT_TEST_PATH
    print(f"Loading Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Data...")

    print("Training dataset:", train_path)
    print("Testing dataset :", test_path)

    train_df = load_file_to_df(train_path)
    test_df = load_file_to_df(test_path)

    train_ds = prepare_multitask_dataset(train_df)
    eval_ds = prepare_multitask_dataset(test_df)

    neutral_ratios = [row["loaded_words_ratio"] for row in train_ds if row["propaganda_label"] == 0]
    propaganda_ratios = [row["loaded_words_ratio"] for row in train_ds if row["propaganda_label"] == 1]

    print("=" * 50)
    print("Neutral records:", len(neutral_ratios))
    if neutral_ratios:
        print("Neutral average loaded ratio:", sum(neutral_ratios) / len(neutral_ratios))
    print("Propaganda records:", len(propaganda_ratios))
    if propaganda_ratios:
        print("Propaganda average loaded ratio:", sum(propaganda_ratios) / len(propaganda_ratios))
    print("=" * 50)

    train_ds = train_ds.map(lambda batch: format_prompts(batch, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda batch: format_prompts(batch, tokenizer), batched=True)

    token_lengths = [len(tokenizer(t).input_ids) for t in train_ds["text"][:200]]
    if token_lengths:
        over_budget = sum(1 for l in token_lengths if l > MAX_SEQ_LENGTH)
        print(f"Sampled sequence lengths - max: {max(token_lengths)}, "
              f"mean: {sum(token_lengths)/len(token_lengths):.1f}, "
              f"over budget ({MAX_SEQ_LENGTH}): {over_budget}/{len(token_lengths)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading Quantized Model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},   
        trust_remote_code=True,
        attn_implementation="sdpa"  
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir=f"./train/checkpoints/{task_name}_model",
        learning_rate=2e-4,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  
        max_grad_norm=0.3,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        logging_steps=10,
        gradient_checkpointing=True,
        save_total_limit=1,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        optim="paged_adamw_8bit",    
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training session on local GPU...")
    trainer.train()

    model = trainer.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join("train","models",f"{task_name}_{timestamp}")   
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Adapters successfully saved to: {save_path}")

    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    print("\nRunning post-training evaluation on test set...")
    model.eval()

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    st_mapping = {"reporting": 0, "opinion": 1}
    pr_mapping = {"neutral": 0, "propaganda": 1}
    at_mapping = {"supported_claim": 0, "unsupported_claim": 1}

    true_st, true_pr, true_at = [], [], []
    pred_st, pred_pr, pred_at = [], [], []
    unparsed_count = 0

    with torch.no_grad():
        for example in tqdm(eval_ds, desc="Evaluating Test Set"):
            messages = build_messages(
                example["title"],
                example["content"],
                example["loaded_words_ratio"],
                include_answer=False
            )
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_EVAL_INPUT_TOKENS
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            st_pred, pr_pred, at_pred = parse_output(decoded_output)
            if "unknown" in (st_pred, pr_pred, at_pred):
                unparsed_count += 1

            true_st_val = example["statement_type_label"]
            true_pr_val = example["propaganda_label"]
            true_at_val = example["attribution_label"]

            true_st.append(true_st_val)
            true_pr.append(true_pr_val)
            true_at.append(true_at_val)

            pred_st.append(st_mapping.get(st_pred, 1 - true_st_val))
            pred_pr.append(pr_mapping.get(pr_pred, 1 - true_pr_val))
            pred_at.append(at_mapping.get(at_pred, 1 - true_at_val))

    st_acc = accuracy_metric.compute(predictions=pred_st, references=true_st)["accuracy"]
    pr_acc = accuracy_metric.compute(predictions=pred_pr, references=true_pr)["accuracy"]
    at_acc = accuracy_metric.compute(predictions=pred_at, references=true_at)["accuracy"]

    pr_f1 = f1_metric.compute(predictions=pred_pr, references=true_pr, average="macro")["f1"]
    avg_accuracy = (st_acc + pr_acc + at_acc) / 3.0
    parse_failure_rate = unparsed_count / max(len(eval_ds), 1)

    print("\n" + "=" * 40)
    print("FINAL LLM TEST SET EVALUATION REPORT")
    print("=" * 40)
    print(f"Statement Type Accuracy: {st_acc:.4f}")
    print(f"Propaganda Accuracy:     {pr_acc:.4f}")
    print(f"Propaganda F1-Macro:     {pr_f1:.4f}")
    print(f"Attribution Accuracy:    {at_acc:.4f}")
    print(f"Parse Failure Rate:      {parse_failure_rate:.4f}")
    print("-" * 40)
    print(f"Average Multi-task Accuracy: {avg_accuracy:.4f}")
    print("=" * 40)
    return {
        "accuracy": avg_accuracy,
        "propaganda_f1": pr_f1,
        "statement_accuracy": st_acc,
        "attribution_accuracy": at_acc,
        "model_path": save_path,
    }

if __name__ == "__main__":
    run_classifier_training(task_name="multitask")