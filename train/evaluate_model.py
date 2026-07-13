from __future__ import annotations
import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

from train.prepare_data import load_file_to_df, prepare_multitask_dataset
from train.prompt_utils import build_messages, parse_output

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_INPUT_TOKENS = 1024

TEST_PATH = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_test.jsonl"))


def run_reevaluation(adapter_path: str, output_report_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_df = load_file_to_df(TEST_PATH)
    eval_ds = prepare_multitask_dataset(test_df)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    st_mapping = {"reporting": 0, "opinion": 1}
    pr_mapping = {"neutral": 0, "propaganda": 1}
    at_mapping = {"supported_claim": 0, "unsupported_claim": 1}

    true_st, true_pr, true_at = [], [], []
    pred_st, pred_pr, pred_at = [], [], []
    unparsed_count = 0

    with torch.no_grad():
        for example in tqdm(eval_ds, desc="Reevaluating Test Set"):
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
                max_length=MAX_INPUT_TOKENS
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

    report = {
        "statement_type_accuracy": st_acc,
        "propaganda_accuracy": pr_acc,
        "propaganda_f1_macro": pr_f1,
        "attribution_accuracy": at_acc,
        "average_accuracy": avg_accuracy,
        "parse_failure_rate": parse_failure_rate,
        "num_examples": len(eval_ds)
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if output_report_path:
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_report", default=None)
    args = parser.parse_args()
    run_reevaluation(args.adapter_path, args.output_report)