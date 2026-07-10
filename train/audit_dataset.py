from __future__ import annotations
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

from train.prepare_data import load_file_to_df, prepare_multitask_dataset

BASE_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MULTITASK_PATH = "./train/models/fine_tuned_llm_multitask"
AUDIT_TARGET_DATASET = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_test.jsonl"))
OUTPUT_CSV_REPORT = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "flagged_errors.csv"))


def parse_clean_output(generation_text: str) -> tuple[str, str, str]:
    try:
        if "التقييم النهائي:" in generation_text:
            parsed = generation_text.split("التقييم النهائي:")[-1].strip().lower()
            parts = [p.strip() for p in parsed.split("|")]
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
    except Exception:
        pass
    return "unknown", "unknown", "unknown"


def run_dataset_audit():
    if not os.path.exists(MULTITASK_PATH):
        raise FileNotFoundError(
            f"An active fine-tuned multitask adapter model is required at {MULTITASK_PATH} to run data auditing."
        )

    print(f"Loading Base: {BASE_MODEL_NAME} with Adapter: {MULTITASK_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, MULTITASK_PATH)
    model.eval()

    print(f"Scanning target dataset: {AUDIT_TARGET_DATASET}")
    test_df = load_file_to_df(AUDIT_TARGET_DATASET)
    eval_ds = prepare_multitask_dataset(test_df)

    reverse_st = {0: "reporting", 1: "opinion"}
    reverse_pr = {0: "neutral", 1: "propaganda"}
    reverse_at = {0: "supported_claim", 1: "unsupported_claim"}

    system_content = (
        "أنت مساعد ذكي متخصص في تصنيف النصوص الإخبارية العربية بدقة. قم بتحليل العنوان والمحتوى الإخباري وصنفهم لثلاثة مهام:\n"
        "1. نوع العبارة (statement_type): إما reporting أو opinion\n"
        "2. البروباغندا (propaganda): إما neutral أو propaganda\n"
        "3. الإسناد (attribution): إما supported_claim أو unsupported_claim\n\n"
        "يجب أن تبدأ إجابتك بخطوة تحليل قصيرة (تحليل النص:)، ثم تكتب التقييم النهائي بالصيغة التالية بالضبط:\n"
        "التقييم النهائي: [نوع العبارة] | [البروباغندا] | [الإسناد]"
    )

    flagged_records = []

    with torch.no_grad():
        for i, example in enumerate(tqdm(eval_ds, desc="Auditing Labels")):
            title = example["title"]
            content = example["content"]

            user_content = f"العنوان: {title}\nالمحتوى: {content}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=150, 
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            raw_gen = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            st_pred, pr_pred, at_pred = parse_clean_output(raw_gen)

            gold_st = reverse_st[example["statement_type_label"]]
            gold_pr = reverse_pr[example["propaganda_label"]]
            gold_at = reverse_at[example["attribution_label"]]

            mismatches = []
            if st_pred != gold_st:
                mismatches.append(f"statement_type ({gold_st} vs {st_pred})")
            if pr_pred != gold_pr:
                mismatches.append(f"propaganda ({gold_pr} vs {pr_pred})")
            if at_pred != gold_at:
                mismatches.append(f"attribution ({gold_at} vs {at_pred})")

            if mismatches:
                reasoning = raw_gen.split("التقييم النهائي:")[0].replace("تحليل النص:", "").strip()
                
                flagged_records.append({
                    "Index": i,
                    "Title": title,
                    "Content_Excerpt": content[:120] + "...",
                    "Mismatch_Tasks": ", ".join(mismatches),
                    "Gold_Statement": gold_st,
                    "Model_Statement": st_pred,
                    "Gold_Propaganda": gold_pr,
                    "Model_Propaganda": pr_pred,
                    "Gold_Attribution": gold_at,
                    "Model_Attribution": at_pred,
                    "Model_Reasoning": reasoning
                })

    if flagged_records:
        audit_df = pd.DataFrame(flagged_records)
        audit_df.to_csv(OUTPUT_CSV_REPORT, index=False, encoding="utf-8-sig")
        print(f"\n[AUDIT COMPLETE] Flagged {len(flagged_records)} potential label errors in dataset.")
        print(f"Discrepancies saved to: {OUTPUT_CSV_REPORT}")
        print("You can now open this spreadsheet, evaluate, and correct the noisy test labels.")
    else:
        print("\n[AUDIT COMPLETE] No labeling discrepancies detected between the model and the test labels.")


if __name__ == "__main__":
    run_dataset_audit()