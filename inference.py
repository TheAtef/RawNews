from __future__ import annotations
import collections
import collections.abc
import json
import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, BertModel
from arabert.preprocess import ArabertPreprocessor  

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from core.config import settings

PROPAGANDA_MAP = {"neutral": 0, "loaded_language": 1, "doubt_casting": 2, "propaganda": 3}
STATEMENT_MAP = {"reporting": 0, "opinion": 1, "speculation": 2}
ATTRIBUTION_MAP = {"supported_claim": 0, "unsupported_claim": 1}

class MultiHeadAraBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 3 Custom Heads
        self.statement_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, len(STATEMENT_MAP))
        )
        self.propaganda_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, len(PROPAGANDA_MAP))
        )
        self.attribution_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, len(ATTRIBUTION_MAP))
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1])
        
        st_logits = self.statement_head(pooled_output)
        pr_logits = self.propaganda_head(pooled_output)
        at_logits = self.attribution_head(pooled_output)
        
        return torch.cat([st_logits, pr_logits, at_logits], dim=-1)

    @classmethod
    def from_pretrained(cls, model_dir):
        config = AutoConfig.from_pretrained(model_dir)
        model = cls(config)
        
        state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Model weights not found at {state_dict_path}")
            
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        
        model.load_state_dict(state_dict)
        return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.abspath(settings.multi_sentiment_model_id)

print(f"Loading custom Multi-Task model from: {model_path}")
model = MultiHeadAraBERT.from_pretrained(model_path)
model.to(device)
model.eval()  # Put model in evaluation mode

tokenizer = AutoTokenizer.from_pretrained(model_path)

config = AutoConfig.from_pretrained(model_path)
arabert_preprocessor = ArabertPreprocessor(model_name=config._name_or_path)

cleaner = ArabicNewsCleaner(remove_numbers=False, keep_quotes=True)
normalizer = ArabicNormalizer()

with open("piece.txt", "r", encoding='utf-8') as file:
    content = file.read()
    
clean = cleaner.clean(content)
normal = normalizer.normalize(clean)
segmented = arabert_preprocessor.preprocess(normal)
print(f"Preprocessed Text for Inference:\n{segmented[:200]}...\n")

reverse_statement_map = {v: k for k, v in STATEMENT_MAP.items()}
reverse_propaganda_map = {v: k for k, v in PROPAGANDA_MAP.items()}
reverse_attribution_map = {v: k for k, v in ATTRIBUTION_MAP.items()}

inputs = tokenizer(segmented, truncation=True, max_length=512, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).squeeze(0)

# Slice logits based on Map Lengths
st_len = len(STATEMENT_MAP)
pr_len = len(PROPAGANDA_MAP)

statement_logits = logits[:st_len]
propaganda_logits = logits[st_len : st_len + pr_len]
attribution_logits = logits[st_len + pr_len :]

# Calculate Probabilities
statement_probs = torch.softmax(statement_logits, dim=-1)
propaganda_probs = torch.softmax(propaganda_logits, dim=-1)
attribution_probs = torch.softmax(attribution_logits, dim=-1)

# Get highest probability index
statement_idx = int(torch.argmax(statement_probs).item())
propaganda_idx = int(torch.argmax(propaganda_probs).item())
attribution_idx = int(torch.argmax(attribution_probs).item())

result = {
    "statement_type_label": {
        "label": reverse_statement_map[statement_idx],
        "score": round(float(statement_probs[statement_idx]), 4),
    },
    "propaganda_label": {
        "label": reverse_propaganda_map[propaganda_idx],
        "score": round(float(propaganda_probs[propaganda_idx]), 4),
    },
    "attribution_label": {
        "label": reverse_attribution_map[attribution_idx],
        "score": round(float(attribution_probs[attribution_idx]), 4),
    },
}

print(json.dumps(result, ensure_ascii=False, indent=2))