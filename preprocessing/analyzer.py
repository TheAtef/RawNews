from __future__ import annotations
import re
from typing import List, Dict, Set
from core.constants import BIAS_INDICATORS, PROPAGANDA_PATTERNS
from core.sources_list import ARABIC_NEWS_SOURCES

SOURCE_AUTHORITY: Dict[str, float] = {
    source.name: source.reliability_score for source in ARABIC_NEWS_SOURCES
}

LOADED_WORDS: Set[str] = set()
for category_dict in BIAS_INDICATORS.values():
    LOADED_WORDS.update(category_dict.keys())

ATTRIBUTION_MARKERS: List[str] = [
    "وفقاً لـ", "حسب بيان", "أعلنت وزارة", "صرح", "أكد", "نقلت عن", "قالت", "ذكر", "أوضح", "أشار"
]


class HeuristicScorer:
    def __init__(self, source_registry: Dict[str, float] = None):
        self.source_registry = source_registry if source_registry is not None else SOURCE_AUTHORITY

    def calculate_source_authority(self, source_name: str) -> float:
        return self.source_registry.get(source_name, 0.40)

    def calculate_neutrality_score(self, tokens: List[str]) -> float:
        if not tokens:
            return 1.0
        
        loaded_count = sum(1 for token in tokens if token in LOADED_WORDS)
        density = loaded_count / len(tokens)
        
        score = max(0.0, 1.0 - (density * 15.0))
        return round(score, 2)

    def calculate_attribution_score(self, text: str) -> float:
        score = 0.0
        quotes = re.findall(r'["«»“”]', text)
        if len(quotes) >= 2:
            score += 0.40            
        marker_matches = sum(1 for marker in ATTRIBUTION_MARKERS if marker in text)
        score += min(0.60, marker_matches * 0.20)
        
        return round(score, 2)

    def evaluate_article(
        self, 
        source_name: str, 
        raw_text: str, 
        tokens: List[str], 
        consensus_score: float = 1.0
    ) -> Dict[str, float]:
        s_auth = self.calculate_source_authority(source_name)
        s_neut = self.calculate_neutrality_score(tokens)
        s_attr = self.calculate_attribution_score(raw_text)
        s_cons = consensus_score

        composite = (0.35 * s_auth) + (0.35 * s_neut) + (0.15 * s_attr) + (0.15 * s_cons)
        return {
            "reliability_score": round(composite, 2),
            "neutrality_score": s_neut,
            "attribution_score": s_attr
        }


class StoryGrouper:
    def __init__(self, overlap_threshold: float = 0.18):
        self.threshold = overlap_threshold
        self.clusters: Dict[int, List[Dict]] = {}
        self._next_cluster_id = 1

    def _get_unique_content_tokens(self, tokens: List[str]) -> Set[str]:
        return {t for t in tokens if len(t) > 3}

    def add_to_cluster(self, article: Dict, tokens: List[str]) -> int:
        art_unique = self._get_unique_content_tokens(tokens)
        if not art_unique:
            cid = self._next_cluster_id
            self.clusters[cid] = [article]
            self._next_cluster_id += 1
            return cid

        best_cid = None
        best_similarity = 0.0

        for cid, cluster_articles in self.clusters.items():
            anchor_tokens = self._get_unique_content_tokens(cluster_articles[0]["tokens"])
            intersection = art_unique.intersection(anchor_tokens)
            union = art_unique.union(anchor_tokens)
            similarity = len(intersection) / len(union) if union else 0.0

            if similarity > self.threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cid = cid

        if best_cid is not None:
            self.clusters[best_cid].append(article)
            return best_cid
        else:
            cid = self._next_cluster_id
            self.clusters[cid] = [article]
            self._next_cluster_id += 1
            return cid
            
    def seed_cluster_anchor(self, cluster_id: int, tokens: List[str]) -> None:
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
        self.clusters[cluster_id].append({"tokens": tokens})
        if cluster_id >= self._next_cluster_id:
            self._next_cluster_id = cluster_id + 1