from __future__ import annotations
import collections
import collections.abc
import json
import re
import sys
import os

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor  

from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from preprocessing.tokenizer import ArabicTokenizer
from core.config import settings
from train.prepare_data import ATTRIBUTION_MAP, PROPAGANDA_MAP, STATEMENT_MAP
model_path = os.path.abspath(settings.multi_sentiment_model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading local model from: {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

arabert_preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")

content = ''
cleaner = ArabicNewsCleaner(remove_numbers=False, keep_quotes=True)
normalizer = ArabicNormalizer()

with open("piece.txt", "r", encoding='utf-8') as file:
    content = file.read()
    
clean = cleaner.clean(content)
normal = normalizer.normalize(clean)

segmented = arabert_preprocessor.preprocess(normal)
print(f"Preprocessed Text for Inference:\n{segmented}\n")

reverse_statement_map = {v: k for k, v in STATEMENT_MAP.items()}
reverse_propaganda_map = {v: k for k, v in PROPAGANDA_MAP.items()}
reverse_attribution_map = {v: k for k, v in ATTRIBUTION_MAP.items()}

inputs = tokenizer(segmented, truncation=True, max_length=256, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits.squeeze(0)

statement_logits = logits[: len(STATEMENT_MAP)]
propaganda_logits = logits[len(STATEMENT_MAP) : len(STATEMENT_MAP) + len(PROPAGANDA_MAP)]
attribution_logits = logits[-len(ATTRIBUTION_MAP) :]

statement_probs = torch.softmax(statement_logits, dim=-1)
propaganda_probs = torch.softmax(propaganda_logits, dim=-1)
attribution_probs = torch.softmax(attribution_logits, dim=-1)

statement_idx = int(torch.argmax(statement_probs).item())
propaganda_idx = int(torch.argmax(propaganda_probs).item())
attribution_idx = int(torch.argmax(attribution_probs).item())

result = {
    "statement_type_label": {
        "label": reverse_statement_map[statement_idx],
        "score": float(statement_probs[statement_idx].detach()),
    },
    "propaganda_label": {
        "label": reverse_propaganda_map[propaganda_idx],
        "score": float(propaganda_probs[propaganda_idx].detach()),
    },
    "attribution_label": {
        "label": reverse_attribution_map[attribution_idx],
        "score": float(attribution_probs[attribution_idx].detach()),
    },
}

print(json.dumps(result, ensure_ascii=False, indent=2))


# import pandas as pd

# df = pd.read_json("output.json")

# print(df["statement_type"].value_counts())

# statement_type
# speculation          2253
# fact                  564
# factual_reporting     526
# opinion               101
# Name: count, dtype: int64