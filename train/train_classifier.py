from __future__ import annotations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import torch
import evaluate
import inspect
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

from train.prepare_data import load_file_to_df, prepare_multitask_dataset


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

TRAIN_PATH = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_train.jsonl"))
TEST_PATH = os.path.abspath(os.path.join(script_dir, "..", "train/clean_data", "relabeled_test.jsonl"))

def format_prompts(batch, tokenizer):
    formatted_texts = []
    for title, content, st, pr, at in zip(
        batch["title"], 
        batch["content"], 
        batch["statement_type_label"], 
        batch["propaganda_label"], 
        batch["attribution_label"]
    ):
        st_str = "reporting" if st == 0 else "opinion"
        pr_str = "neutral" if pr == 0 else "propaganda"
        at_str = "supported_claim" if at == 0 else "unsupported_claim"
        
        st_reason = "النص يركز على نقل وقائع وتفاصيل ملموسة بنبرة حيادية وموضوعية خالية من الانحياز الشخصي." if st == 0 else "يتضمن النص تعبيرات تدل على التقييم الشخصي، التخمين، أو الرؤية الذاتية للكاتب."
        pr_reason = "لغة النص تلتزم بالمعايير المهنية والصحفية، وتخلو من الأساليب البلاغية العاطفية أو التشكيك غير المبرر." if pr == 0 else "يلاحظ استخدام صياغات عاطفية، أو محاولات لتوجيه الرأي العام، أو إطلاق أحكام مسبقة."
        at_reason = "الادعاءات الواردة في النص يتم إسنادها بوضوح إلى مصادر محددة، وثائق، أو جهات مسؤولة تدعم صحتها." if at == 0 else "الادعاءات تُطرح بشكل مرسل دون الإشارة إلى مصادر واضحة، شهادات موثوقة، أو أدلة تدعمها."
        
        system_content = (
            "أنت مساعد ذكي متخصص في تصنيف النصوص الإخبارية العربية بدقة. قم بتحليل العنوان والمحتوى الإخباري وصنفهم لثلاثة مهام:\n"
            "1. نوع العبارة (statement_type): إما reporting أو opinion\n"
            "2. البروباغندا (propaganda): إما neutral أو propaganda\n"
            "3. الإسناد (attribution): إما supported_claim أو unsupported_claim\n\n"
            "يجب أن تبدأ إجابتك بخطوة تحليل قصيرة (تحليل النص:)، ثم تكتب التقييم النهائي بالصيغة التالية بالضبط:\n"
            "التقييم النهائي: [نوع العبارة] | [البروباغندا] | [الإسناد]"
        )
        
        user_content = f"العنوان: {title}\nالمحتوى: {content}"
        
        assistant_content = (
            f"تحليل النص:\n"
            f"1. نوع العبارة: {st_reason}\n"
            f"2. البروباغندا: {pr_reason}\n"
            f"3. الإسناد: {at_reason}\n\n"
            f"التقييم النهائي: {st_str} | {pr_str} | {at_str}"
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append(text)
        
    return {"text": formatted_texts}

def parse_output(generation_text: str):
    try:
        if "التقييم النهائي:" in generation_text:
            parsed = generation_text.split("التقييم النهائي:")[-1].strip().lower()
            parts = [p.strip() for p in parsed.split("|")]
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
    except Exception:
        pass
    return "unknown", "unknown", "unknown"

def run_classifier_training(task_name: str = "multitask"):
    print(f"Loading Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Data...")
    train_df = load_file_to_df(TRAIN_PATH)
    test_df = load_file_to_df(TEST_PATH)

    train_ds = prepare_multitask_dataset(train_df)
    eval_ds = prepare_multitask_dataset(test_df)

    train_ds = train_ds.map(lambda batch: format_prompts(batch, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda batch: format_prompts(batch, tokenizer), batched=True)

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
        device_map="auto",
        trust_remote_code=True
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
        num_train_epochs=3,             
        per_device_train_batch_size=2,    
        per_device_eval_batch_size=2,      
        gradient_accumulation_steps=8, 
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
        max_length=512, 
        loss_type="chunked_nll",
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

    save_path = f"./train/models/fine_tuned_llm_{task_name}"
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

    system_content = (
        "أنت مساعد ذكي متخصص في تصنيف النصوص الإخبارية العربية بدقة. قم بتحليل العنوان والمحتوى الإخباري وصنفهم لثلاثة مهام:\n"
        "1. نوع العبارة (statement_type): إما reporting أو opinion\n"
        "2. البروباغندا (propaganda): إما neutral أو propaganda\n"
        "3. الإسناد (attribution): إما supported_claim أو unsupported_claim\n\n"
        "أجب فقط بالصيغة التالية بالضبط دون أي مقدمات أو شرح إضافي:\n"
        "[نوع العبارة] | [البروباغندا] | [الإسناد]"
    )

    with torch.no_grad():
        for example in tqdm(eval_ds, desc="Evaluating Test Set"):
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
                max_new_tokens=24,
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            st_pred, pr_pred, at_pred = parse_output(decoded_output)

            true_st.append(example["statement_type_label"])
            true_pr.append(example["propaganda_label"])
            true_at.append(example["attribution_label"])

            pred_st.append(st_mapping.get(st_pred, 0))
            pred_pr.append(pr_mapping.get(pr_pred, 0))
            pred_at.append(at_mapping.get(at_pred, 0))

    st_acc = accuracy_metric.compute(predictions=pred_st, references=true_st)["accuracy"]
    pr_acc = accuracy_metric.compute(predictions=pred_pr, references=true_pr)["accuracy"]
    at_acc = accuracy_metric.compute(predictions=pred_at, references=true_at)["accuracy"]
    
    pr_f1 = f1_metric.compute(predictions=pred_pr, references=true_pr, average="macro")["f1"]
    avg_accuracy = (st_acc + pr_acc + at_acc) / 3.0

    print("\n" + "="*40)
    print("FINAL LLM TEST SET EVALUATION REPORT")
    print("="*40)
    print(f"Statement Type Accuracy: {st_acc:.4f}")
    print(f"Propaganda Accuracy:     {pr_acc:.4f}")
    print(f"Propaganda F1-Macro:     {pr_f1:.4f}")
    print(f"Attribution Accuracy:    {at_acc:.4f}")
    print("-"*40)
    print(f"Average Multi-task Accuracy: {avg_accuracy:.4f}")
    print("="*40)


if __name__ == "__main__":
    run_classifier_training(task_name="multitask")