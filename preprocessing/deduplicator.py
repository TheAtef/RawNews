from __future__ import annotations
import hashlib
import re
from typing import FrozenSet, List, Optional, Set, Tuple, Dict

_RE_WS = re.compile(r"\s+")

def _fingerprint(text: str) -> str:
    normalized = _RE_WS.sub(" ", text).strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def _word_shingles(text: str, n: int = 3) -> FrozenSet[str]:
    normalized = _RE_WS.sub(" ", text).strip().lower()
    words = normalized.split()
    if len(words) < n:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    union_size = len(a | b)
    if union_size == 0:
        return 1.0
    return len(a & b) / union_size

class ArticleDeduplicator:
    def __init__(self, similarity_threshold: float = 0.82, shingle_size: int = 3) -> None:
        self._threshold = similarity_threshold
        self._n = shingle_size
        self._seen_fingerprints: Set[str] = set()
        self._seen_shingles: List[Tuple[str, FrozenSet[str]]] = []
        self._inverted_index: Dict[str, Set[str]] = {}

    def is_duplicate(self, article_id: str, text: str) -> Tuple[bool, Optional[str]]:
        if not text or not text.strip():
            return False, None

        fp = _fingerprint(text)
        if fp in self._seen_fingerprints:
            return True, "exact"

        shingles = _word_shingles(text, self._n)
        
        candidate_ids: Set[str] = set()
        for shingle in shingles:
            if shingle in self._inverted_index:
                candidate_ids.update(self._inverted_index[shingle])

        for seen_id, seen_sh in self._seen_shingles:
            if seen_id not in candidate_ids:
                continue
                
            similarity = _jaccard(shingles, seen_sh)
            if similarity >= self._threshold:
                return True, "near"

        self._seen_fingerprints.add(fp)
        self._seen_shingles.append((article_id, shingles))
        
        for shingle in shingles:
            if shingle not in self._inverted_index:
                self._inverted_index[shingle] = set()
            self._inverted_index[shingle].add(article_id)

        return False, None

    def add_known_fingerprint(self, text: str) -> None:
        fp = _fingerprint(text)
        self._seen_fingerprints.add(fp)

    def seed_known_article(self, article_id: str, text: str) -> None:
        if not text or not text.strip():
            return
        fp = _fingerprint(text)
        self._seen_fingerprints.add(fp)
        shingles = _word_shingles(text, self._n)
        self._seen_shingles.append((article_id, shingles))
        for shingle in shingles:
            if shingle not in self._inverted_index:
                self._inverted_index[shingle] = set()
            self._inverted_index[shingle].add(article_id)

    def reset(self) -> None:
        self._seen_fingerprints.clear()
        self._seen_shingles.clear()
        self._inverted_index.clear()