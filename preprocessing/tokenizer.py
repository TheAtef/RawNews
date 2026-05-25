from __future__ import annotations
import re
from typing import List, Optional, Set, FrozenSet

_RE_MIXED_TOKEN = re.compile(
    r"[a-zA-Z0-9\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+"
)

_BIAS_SENSITIVE_STOPWORDS: FrozenSet[str] = frozenset({
    "هو", "هي", "هم", "هن", "أنا", "أنت", "أنتم", "نحن", "هذا", "هذه",
    "ذلك", "تلك", "هؤلاء", "أولئك",
    "في", "من", "إلى", "على", "عن", "مع", "بين", "بعد", "قبل", "عند",
    "حتى", "حول", "ضد", "خلال", "منذ", "لدى", "نحو", "أمام", "وراء",
    "تحت", "فوق", "بجانب",
    "و", "أو", "بل", "ثم", "إذ", "إذا", "لو", "لأن", "حيث", "كما", "كيف",
    "ال", "لل", "بال", "وال", "فال",
    "الذي", "التي", "الذين", "اللواتي", "اللاتي",
    "أين", "متى", "لماذا", "هل",
    "أفاد", "أوضح", "أشار", "ذكر", "قال", "أضاف", "أكد"
})

class ArabicTokenizer:
    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return _RE_MIXED_TOKEN.findall(text)

class ArabicStopwordFilter:
    def __init__(self, extra_stopwords: Optional[List[str]] = None) -> None:
        combined: Set[str] = set(_BIAS_SENSITIVE_STOPWORDS)
        if extra_stopwords:
            combined.update(extra_stopwords)
        self._stopwords: FrozenSet[str] = frozenset(combined)

    def filter(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self._stopwords]