from __future__ import annotations
from typing import List, Set
from core.constants import BIAS_INDICATORS
from preprocessing.normalizer import ArabicNormalizer
NORMALIZER = ArabicNormalizer()


LOADED_PHRASES: Set[str] = set()

for category_dict in BIAS_INDICATORS.values():
    for phrase in category_dict.keys():
        normalized_phrase = NORMALIZER.normalize(phrase).strip()

        if normalized_phrase:
            LOADED_PHRASES.add(normalized_phrase)



def calculate_loaded_words_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0

    normalized_tokens = [ NORMALIZER.normalize(token).strip() for token in tokens ]

    normalized_tokens = [token for token in normalized_tokens if token ]

    if not normalized_tokens:
        return 0.0

    phrase_tokens_list = [phrase.split() for phrase in LOADED_PHRASES  if phrase.strip() ]

    phrase_tokens_list.sort( key=len, reverse=True )

    total_loaded_tokens = 0
    i = 0

    while i < len(normalized_tokens):
        matched = False

        for phrase_tokens in phrase_tokens_list:
            phrase_length = len(phrase_tokens)

            window = normalized_tokens[i:i + phrase_length]

            if window == phrase_tokens:
                total_loaded_tokens += phrase_length
                i += phrase_length
                matched = True
                break

        if not matched:
            i += 1

    return total_loaded_tokens / len(normalized_tokens)
