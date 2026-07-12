import os
import logging
import json
import re
import pandas as pd
from datasets import Dataset, load_dataset
from arabert.preprocess import ArabertPreprocessor
from typing import List, Dict, Any

from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from preprocessing.tokenizer import ArabicTokenizer, ArabicStopwordFilter
from preprocessing.propaganda_features import calculate_loaded_words_ratio
from preprocessing.ner import NER
from core.constants import PROPAGANDA_PATTERNS, BIAS_INDICATORS, ARABIC_BOILERPLATE_KEYWORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_CONTENT_WORDS = 200

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

NUM_CLASSES = {
    "statement_type": len(set(STATEMENT_MAP.values())),
    "propaganda": len(set(PROPAGANDA_MAP.values())),
    "attribution": len(set(ATTRIBUTION_MAP.values()))
}

REVERSE_MAPS = {
    "propaganda": {0: "neutral", 1: "propaganda"},
    "statement_type": {0: "reporting", 1: "opinion"},
    "attribution": {0: "supported_claim", 1: "unsupported_claim"}
}

GLOBAL_CLEANER = ArabicNewsCleaner()
GLOBAL_NORMALIZER = ArabicNormalizer()
GLOBAL_TOKENIZER = ArabicTokenizer()
GLOBAL_STOP_FILTER = ArabicStopwordFilter()
GLOBAL_NER = NER()

def clean_and_strip_boilerplate(text: str) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.؟!])\s+', text)
    cleaned_sentences = []
    for sent in sentences:
        match_count = sum(1 for kw in ARABIC_BOILERPLATE_KEYWORDS if kw in sent)
        if match_count < 2:
            cleaned_sentences.append(sent)
    return " ".join(cleaned_sentences)

def count_subjective_markers(text: str) -> int:
    subjective_words = ["أنا", "نحن", "أرى", "أعتقد", "نعتقد", "نرى", "في رأيي", "برأيي"]
    count = 0
    for word in subjective_words:
        count += len(re.findall(r'\b' + re.escape(word) + r'\b', text))
    return count

def get_entity_features(text: str) -> Dict[str, int]:
    try:
        entities = GLOBAL_NER.extract_entities(text)
        total_entities = (
            len(entities.get("person", [])) + 
            len(entities.get("location", [])) + 
            len(entities.get("organization", []))
        )
        return {
            "entity_count": total_entities,
            "unique_orgs_count": len(entities.get("organization", []))
        }
    except Exception:
        return {"entity_count": 0, "unique_orgs_count": 0}

def sentence_aware_head_tail(text: str, max_words: int = 200) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.؟!])\s+', text)
    word_count = 0
    head_sentences = []
    
    for sent in sentences:
        sent_words = len(sent.split())
        if word_count + sent_words <= max_words // 2:
            head_sentences.append(sent)
            word_count += sent_words
        else:
            break
            
    tail_sentences = []
    tail_word_count = 0
    for sent in reversed(sentences):
        sent_words = len(sent.split())
        if tail_word_count + sent_words <= max_words // 2:
            tail_sentences.insert(0, sent)
            tail_word_count += sent_words
        else:
            break
            
    if not head_sentences and not tail_sentences:
        return " ".join(text.split()[:max_words])
        
    return " ".join(head_sentences) + " ... " + " ".join(tail_sentences)

def extract_advanced_features(title: str, content: str, tokens: List[str]) -> Dict[str, Any]:
    full_text = f"{title} {content}"
    normalized_tokens = [GLOBAL_NORMALIZER.normalize(t).strip() for t in tokens if t.strip()]
    total_tokens = len(normalized_tokens) or 1

    quotes_count = len(re.findall(r'["«»“”]', full_text))
    attribution_markers = ["وفقاً لـ", "حسب بيان", "أعلنت وزارة", "صرح", "أكد", "نقلت عن", "قالت", "ذكر", "أوضح", "أشار"]
    markers_found = sum(1 for marker in attribution_markers if marker in full_text)

    detected_patterns = []
    for p in PROPAGANDA_PATTERNS:
        if re.search(p["pattern"], full_text):
            detected_patterns.append(p["type"])

    bias_breakdown = {}
    for category, words_dict in BIAS_INDICATORS.items():
        normalized_keys = {GLOBAL_NORMALIZER.normalize(k).strip() for k in words_dict.keys()}
        matched_count = sum(1 for t in normalized_tokens if t in normalized_keys)
        bias_breakdown[category] = matched_count / total_tokens

    entity_feats = get_entity_features(content)
    subjective_count = count_subjective_markers(full_text)

    return {
        "quotes_count": quotes_count,
        "markers_count": markers_found,
        "detected_patterns": list(set(detected_patterns)),
        "bias_breakdown": bias_breakdown,
        "entity_count": entity_feats["entity_count"],
        "unique_orgs_count": entity_feats["unique_orgs_count"],
        "subjective_markers_count": subjective_count
    }

def bootstrap_armpro_single_file(file_path: str):
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    try:
        ds = load_dataset("QCRI/ArmPro", "binary")
    except Exception as e:
        logger.error(f"Could not load 'QCRI/ArmPro' (binary config) from Hugging Face: {e}")
        raise e

    available_splits = list(ds.keys())
    file_name = os.path.basename(file_path).lower()
    
    if "test" in file_name or "eval" in file_name:
        hf_split = "test" if "test" in available_splits else available_splits[-1]
    elif "val" in file_name or "dev" in file_name:
        hf_split = "dev" if "dev" in available_splits else ("validation" if "validation" in available_splits else available_splits[-1])
    else:
        hf_split = "train" if "train" in available_splits else available_splits[0]

    data_split = ds[hf_split]
    logger.info(f"Mapping split '{hf_split}' onto {file_path}...")

    possible_text_cols = ["paragraph", "text", "content", "sentence"]
    text_col = next((c for c in possible_text_cols if c in data_split.column_names), data_split.column_names[0])
    
    possible_label_cols = ["label", "coarse_label", "propaganda_label", "class", "is_propaganda"]
    label_col = next((c for c in possible_label_cols if c in data_split.column_names), None)

    records = []
    for row in data_split:
        raw_text = str(row.get(text_col) or "").strip()
        if not raw_text:
            continue
            
        raw_label = row.get(label_col) if label_col else None
        val_str = str(raw_label).lower().strip()
        
        if "non-propagandistic" in val_str or val_str in ["false", "0", "neutral", "no", "non_propaganda"]:
            mapped_propaganda = "neutral"
        elif "propagandistic" in val_str or val_str in ["true", "1", "yes", "propaganda"]:
            mapped_propaganda = "propaganda"
        else:
            mapped_propaganda = "neutral"

        if mapped_propaganda == "propaganda":
            mapped_statement = "opinion"
            mapped_attribution = "unsupported_claim"
        else:
            mapped_statement = "reporting"
            mapped_attribution = "supported_claim"

        records.append({
            "text": raw_text,
            "propaganda_label": mapped_propaganda,
            "statement_type": mapped_statement,
            "attribution_label": mapped_attribution
        })

    logger.info(f"Writing {len(records)} mapped records to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_file_to_df(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        logger.info(f"Target dataset file was not found at: {file_path}. Initiating automatic fallback/bootstrap from QCRI/ArmPro...")
        try:
            bootstrap_armpro_single_file(file_path)
        except Exception as e:
            logger.error(f"Failed to automatically bootstrap from QCRI/ArmPro: {e}")
            raise FileNotFoundError(f"Target dataset file was not found at: {file_path}")

    if file_path.endswith(".jsonl"):
        return pd.read_json(file_path, lines=True, convert_dates=False)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path, convert_dates=False)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")

def _preprocess_sequence_split(row: pd.Series, preprocessor: ArabertPreprocessor = None) -> tuple[str, str]:
    title = ""
    content = ""

    if "optimized_text" in row and pd.notna(row["optimized_text"]):
        parts = str(row["optimized_text"]).split(" [SEP] ")
        if len(parts) == 2:
            title = parts[0]
            content = parts[1]
        else:
            content = str(row["optimized_text"])
    else:
        raw_title = str(row.get("title") or "").strip()
        raw_content = str(row.get("text") or row.get("content") or "").strip()
        title = raw_title if raw_title else ""
        content = raw_content if raw_content else ""

    if preprocessor is not None:
        title = preprocessor.preprocess(title) if title else ""
        content = preprocessor.preprocess(content) if content else ""

    content = sentence_aware_head_tail(content, max_words=MAX_CONTENT_WORDS)

    if not title.strip():
        title = "بدون عنوان"

    return title, content

def _preprocess_sequence(row: pd.Series, preprocessor: ArabertPreprocessor = None) -> str:
    title, content = _preprocess_sequence_split(row, preprocessor)
    if title and content:
        return f"{title} [SEP] {content}"
    return title or content

def prepare_single_dataset(
    df: pd.DataFrame,
    target_task: str,
    preprocessor: ArabertPreprocessor = None
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
    preprocessor: ArabertPreprocessor = None
) -> Dataset:
    processed_records = []

    for idx, row in df.iterrows():
        title, content = _preprocess_sequence_split(row, preprocessor)
        if not content.strip():
            continue

        content = clean_and_strip_boilerplate(content)
        
        cleaned_title = GLOBAL_CLEANER.clean(title)
        normalized_title = GLOBAL_NORMALIZER.normalize(cleaned_title)

        cleaned_content = GLOBAL_CLEANER.clean(content)
        normalized_content = GLOBAL_NORMALIZER.normalize(cleaned_content)

        feature_text = f"{normalized_title} {normalized_content}"
        feature_tokens = GLOBAL_TOKENIZER.tokenize(feature_text)
        filtered_tokens = GLOBAL_STOP_FILTER.filter(feature_tokens)

        features = extract_advanced_features(normalized_title, normalized_content, filtered_tokens)
        loaded_words_ratio = calculate_loaded_words_ratio(filtered_tokens)

        raw_statement = row.get("statement_type")
        raw_propaganda = row.get("propaganda_label")
        raw_attribution = row.get("attribution_label")

        if (raw_statement not in STATEMENT_MAP or
            raw_propaganda not in PROPAGANDA_MAP or
            raw_attribution not in ATTRIBUTION_MAP):
            continue

        processed_records.append({
            "title": normalized_title,
            "content": normalized_content,
            "loaded_words_ratio": loaded_words_ratio,
            "features_json": json.dumps(features, ensure_ascii=False),
            "statement_type_label": STATEMENT_MAP[raw_statement],
            "propaganda_label": PROPAGANDA_MAP[raw_propaganda],
            "attribution_label": ATTRIBUTION_MAP[raw_attribution],
        })

    return Dataset.from_list(processed_records)