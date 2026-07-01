from __future__ import annotations
import torch
import numpy as np
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from train.train_classifier import MultiHeadAraBERT

PROPAGANDA_MAP = {
    "neutral": 0, "loaded_language": 1, "propaganda": 2, 
    "sensationalism": 3, "false_dichotomy": 4, "fear_appeal": 5, 
    "doubt_casting": 6, "exaggeration": 7, "stereotyping": 8
}
STATEMENT_MAP = {
    "fact": 0, "opinion": 1, "speculation": 2, "reporting": 3, "factual_reporting": 4
}
ATTRIBUTION_MAP = {
    "supported_claim": 0, "unsupported_claim": 1, "quote_present": 2, "direct_source": 3
}

REV_PROPAGANDA_MAP = {v: k for k, v in PROPAGANDA_MAP.items()}
REV_STATEMENT_MAP = {v: k for k, v in STATEMENT_MAP.items()}
REV_ATTRIBUTION_MAP = {v: k for k, v in ATTRIBUTION_MAP.items()}


def run_inference(text: str, model_path: str = "./train/models/fine_tuned_arabert_multitask"):
    preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MultiHeadAraBERT.from_pretrained(model_path)
    model.eval()
    
    preprocessed_text = preprocessor.preprocess(text)
    print(f"Preprocessed Text for Inference: {preprocessed_text}\n")
    
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.squeeze(0).cpu()
    
    st_len = len(STATEMENT_MAP)
    pr_len = len(PROPAGANDA_MAP)
    
    statement_logits = logits[:st_len]
    propaganda_logits = logits[st_len : st_len + pr_len]
    attribution_logits = logits[st_len + pr_len :]
    
    statement_probs = torch.softmax(statement_logits, dim=-1).numpy()
    propaganda_probs = torch.softmax(propaganda_logits, dim=-1).numpy()
    attribution_probs = torch.softmax(attribution_logits, dim=-1).numpy()
    
    statement_idx = int(np.argmax(statement_probs))
    propaganda_idx = int(np.argmax(propaganda_probs))
    attribution_idx = int(np.argmax(attribution_probs))
    
    result = {
        "statement_type_label": {
            "label": REV_STATEMENT_MAP[statement_idx],
            "score": float(statement_probs[statement_idx])
        },
        "propaganda_label": {
            "label": REV_PROPAGANDA_MAP[propaganda_idx],
            "score": float(propaganda_probs[propaganda_idx])
        },
        "attribution_label": {
            "label": REV_ATTRIBUTION_MAP[attribution_idx],
            "score": float(attribution_probs[attribution_idx])
        }
    }
    
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_sentence = "اكره احمد الشرع الافارقة وفق ما قالته رويترز"
    run_inference(test_sentence)