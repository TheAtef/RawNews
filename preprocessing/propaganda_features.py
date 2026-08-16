from __future__ import annotations
from typing import List, Dict, Tuple
from core.constants import BIAS_INDICATORS
from preprocessing.normalizer import ArabicNormalizer

NORMALIZER = ArabicNormalizer()

def _light_stem(word: str) -> str:
    word = word.replace("ة", "ه").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    for prefix in ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ب", "ف"]:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            word = word[len(prefix):]
            break
    for suffix in ["ين", "ون", "ات", "يه", "ية", "ها", "هم", "نا", "ي"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            word = word[:-len(suffix)]
            break
    return word

LOADED_PHRASES_WEIGHTS: Dict[Tuple[str, ...], float] = {}

for category, category_dict in BIAS_INDICATORS.items():
    for phrase, weight in category_dict.items():
        normalized_phrase = NORMALIZER.normalize(phrase).strip()
        if normalized_phrase:
            stemmed_tuple = tuple(_light_stem(w) for w in normalized_phrase.split())
            if stemmed_tuple:
                current = LOADED_PHRASES_WEIGHTS.get(stemmed_tuple, 0.0)
                LOADED_PHRASES_WEIGHTS[stemmed_tuple] = max(current, weight)

PHRASE_TUPLES_LIST = sorted(LOADED_PHRASES_WEIGHTS.keys(), key=len, reverse=True)

def calculate_loaded_words_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0

    stemmed_tokens = []
    for token in tokens:
        norm_token = NORMALIZER.normalize(token).strip()
        if norm_token:
            stemmed_tokens.append(_light_stem(norm_token))

    if not stemmed_tokens:
        return 0.0

    total_bias_weight = 0.0
    i = 0

    while i < len(stemmed_tokens):
        matched = False
        for phrase_tuple in PHRASE_TUPLES_LIST:
            phrase_length = len(phrase_tuple)
            window = tuple(stemmed_tokens[i:i + phrase_length])

            if window == phrase_tuple:
                total_bias_weight += LOADED_PHRASES_WEIGHTS[phrase_tuple]
                i += phrase_length
                matched = True
                break

        if not matched:
            i += 1

    return total_bias_weight / len(stemmed_tokens)