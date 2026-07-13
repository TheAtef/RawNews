import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
ADAPTER_PATH = "./train/models/fine_tuned_llm_multitask"

def load_inference_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model

def test_article(title, content, tokenizer, model):
    system_content = (
        "أنت مساعد ذكي متخصص في تصنيف النصوص الإخبارية العربية بدقة. قم بتحليل العنوان والمحتوى الإخباري وصنفهم لثلاثة مهام:\n"
        "1. نوع العبارة (statement_type): إما reporting أو opinion\n"
        "2. البروباغندا (propaganda): إما neutral أو propaganda\n"
        "3. الإسناد (attribution): إما supported_claim أو unsupported_claim\n\n"
        "يجب أن تبدأ إجابتك بخطوة تحليل قصيرة (تحليل النص:)، ثم تكتب التقييم النهائي بالصيغة التالية بالضبط:\n"
        "التقييم النهائي: [نوع العبارة] | [البروباغندا] | [الإسناد]"
    )
    
    user_content = f"العنوان: {title}\nالمحتوى: {content}"
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded_output

if __name__ == "__main__":
    print("Loading model and adapters...")
    tokenizer, model = load_inference_model()
    
    sample_title = "سوريا"
    sample_content = "ذكرت صحيفة رويترز ان الشرع سوف يقوم بزيارة سوريا "    
    print("\nTesting sample article...")
    output = test_article(sample_title, sample_content, tokenizer, model)
    print("\nModel Output:")
    print("-" * 50)
    print(output)
    print("-" * 50)