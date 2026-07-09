from __future__ import annotations
import os
import logging
import pandas as pd
from datasets import Dataset
from arabert.preprocess import ArabertPreprocessor

########################################
from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from preprocessing.tokenizer import ArabicTokenizer, ArabicStopwordFilter
from preprocessing.propaganda_features import calculate_loaded_words_ratio
############################################

logger = logging.getLogger(__name__)

PROPAGANDA_MAP = {
    "neutral": 0, 
    "loaded_language": 1, 
    "doubt_casting": 1, 
    "propaganda": 1
}

STATEMENT_MAP = {
    "reporting": 0, 
    "opinion": 1, 
    "speculation": 1
}

ATTRIBUTION_MAP = {
    "supported_claim": 0, 
    "unsupported_claim": 1
}

GLOBAL_PREPROCESSOR = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")

########################################
GLOBAL_CLEANER = ArabicNewsCleaner()
GLOBAL_NORMALIZER = ArabicNormalizer()
GLOBAL_TOKENIZER = ArabicTokenizer()
GLOBAL_STOP_FILTER = ArabicStopwordFilter()
#############################################


def load_file_to_df(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target dataset file was not found at: {file_path}")
        
    if file_path.endswith(".jsonl"):
        return pd.read_json(file_path, lines=True, convert_dates=False)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path, convert_dates=False)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")


def _extract_head_tail(text: str, max_words: int = 350) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    half = max_words // 2
    return " ".join(words[:half]) + " ... " + " ".join(words[-half:])


def _preprocess_sequence_split(row: pd.Series, preprocessor: ArabertPreprocessor) -> tuple[str, str]:
    title = ""
    content = ""
    
    if "optimized_text" in row and pd.notna(row["optimized_text"]):
        parts = str(row["optimized_text"]).split(" [SEP] ")
        if len(parts) == 2:
            title = preprocessor.preprocess(parts[0])
            content = preprocessor.preprocess(parts[1])
        else:
            content = preprocessor.preprocess(str(row["optimized_text"]))
    else:
        raw_title = str(row.get("title") or "").strip()
        raw_content = str(row.get("text") or row.get("content") or "").strip()
        title = preprocessor.preprocess(raw_title) if raw_title else ""
        content = preprocessor.preprocess(raw_content) if raw_content else ""

    content = _extract_head_tail(content, max_words=350)
    
    if not title.strip():
        title = "بدون عنوان"
        
    return title, content


def _preprocess_sequence(row: pd.Series, preprocessor: ArabertPreprocessor) -> str:
    title, content = _preprocess_sequence_split(row, preprocessor)
    if title and content:
        return f"{title} [SEP] {content}"
    return title or content


def prepare_single_dataset(
    df: pd.DataFrame, 
    target_task: str, 
    preprocessor: ArabertPreprocessor = GLOBAL_PREPROCESSOR
) -> Dataset:
    processed_records = []
    
    for idx, row in df.iterrows():
        cleaned_text = _preprocess_sequence(row, preprocessor)
        if not cleaned_text.strip():
            continue

        if target_task == "propaganda":
            raw_label = row.get("propaganda_label")
            if raw_label not in PROPAGANDA_MAP:
                logger.warning(f"Row {idx}: Invalid propaganda label '{raw_label}'. Row skipped.")
                continue
            label = PROPAGANDA_MAP[raw_label]
            
        elif target_task == "statement_type":
            raw_label = row.get("statement_type")
            if raw_label not in STATEMENT_MAP:
                logger.warning(f"Row {idx}: Invalid statement label '{raw_label}'. Row skipped.")
                continue
            label = STATEMENT_MAP[raw_label]
            
        elif target_task == "attribution":
            raw_label = row.get("attribution_label")
            if raw_label not in ATTRIBUTION_MAP:
                logger.warning(f"Row {idx}: Invalid attribution label '{raw_label}'. Row skipped.")
                continue
            label = ATTRIBUTION_MAP[raw_label]
        else:
            raise ValueError(f"Unknown target task: {target_task}")
            
        processed_records.append({
            "text": cleaned_text,
            "label": label
        })
        
    return Dataset.from_list(processed_records)


def prepare_multitask_dataset(
    df: pd.DataFrame, 
    preprocessor: ArabertPreprocessor = GLOBAL_PREPROCESSOR
) -> Dataset:
    processed_records = []
    
    for idx, row in df.iterrows():
        title, content = _preprocess_sequence_split(row, preprocessor)
        if not content.strip():
            continue
##############################################
        # raw_title = str(row.get("title") or "")
        # raw_content = str(row.get("text") or row.get("content") or "")

        feature_text = f"{title} {content}"

        cleaned_feature_text = GLOBAL_CLEANER.clean(feature_text)
        normalized_feature_text = GLOBAL_NORMALIZER.normalize(cleaned_feature_text)

        feature_tokens = GLOBAL_TOKENIZER.tokenize(normalized_feature_text)

        loaded_words_ratio = calculate_loaded_words_ratio( feature_tokens)
###################################
        raw_statement = row.get("statement_type")
        raw_propaganda = row.get("propaganda_label")
        raw_attribution = row.get("attribution_label")

        if (raw_statement not in STATEMENT_MAP or 
            raw_propaganda not in PROPAGANDA_MAP or 
            raw_attribution not in ATTRIBUTION_MAP):
            continue

        processed_records.append({
            "title": title,
            "content": content,
            "loaded_words_ratio":loaded_words_ratio,
            "statement_type_label": STATEMENT_MAP[raw_statement],
            "propaganda_label": PROPAGANDA_MAP[raw_propaganda],
            "attribution_label": ATTRIBUTION_MAP[raw_attribution],

        })

    return Dataset.from_list(processed_records)