from __future__ import annotations
import re
import torch
import numpy as np
import structlog
from typing import List, Dict, Set, Optional
from transformers import pipeline, AutoTokenizer, AutoModel
from core.config import settings
from core.constants import BIAS_INDICATORS
from core.sources_list import ARABIC_NEWS_SOURCES

logger = structlog.get_logger(__name__)

SOURCE_AUTHORITY: Dict[str, float] = {
    source.name: source.reliability_score for source in ARABIC_NEWS_SOURCES
}

LOADED_WORDS: Set[str] = set()
for category_dict in BIAS_INDICATORS.values():
    LOADED_WORDS.update(category_dict.keys())

ATTRIBUTION_MARKERS: List[str] = [
    "وفقاً لـ", "حسب بيان", "أعلنت وزارة", "صرح", "أكد", "نقلت عن", "قالت", "ذكر", "أوضح", "أشار"
]


class AraBERTClassifier:
    def __init__(self) -> None:
        self.device = 0 if settings.device == "cuda" and torch.cuda.is_available() else -1            
        try:
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=settings.sentiment_model_id,
                device=self.device
            )
            self.enabled = True
        except Exception as e:
            logger.error("arabert_classifier_init_failed", error=str(e))
            self.enabled = False

    def classify(self, text: str, title: str, clean_tokens: List[str]) -> Dict[str, str]:
        if not self.enabled or not text.strip():
            return {
                "propaganda_label": "neutral",
                "statement_type": "fact",
                "attribution_label": "unsupported_claim"
            }

        truncated_text = text[:1000]

        try:
            #propaganda & bias detection using arabert sentiment Analysis
            sent_res = self.sentiment_pipe(truncated_text)[0]
            label = sent_res["label"].upper()  
            score = sent_res["score"]

            if label == "NEG" and score > 0.75:
                propaganda_label = "loaded_language"
            elif label == "POS" and score > 0.85:
                propaganda_label = "propaganda"
            else:
                propaganda_label = "neutral"

            has_speculative_indicators = any(
                w in truncated_text for w in ["قد", "ربما", "يتوقع", "يُحتمل", "سيناريو", "تقديرات"]
            )
            has_opinion_indicators = any(
                w in truncated_text for w in ["أعتقد", "يرى الكاتب", "في رأيي", "وجهة نظر", "أظن"]
            )

            if has_speculative_indicators:
                statement_type = "speculation"
            elif has_opinion_indicators:
                statement_type = "opinion"
            else:
                statement_type = "fact"

            # attribution level classification (supported vs unsupported claim)
            # checks for direct quotations or active linguistic markers of attribution
            has_markers = any(marker in truncated_text for marker in ATTRIBUTION_MARKERS)
            has_quotes = len(re.findall(r'["«»“”]', truncated_text)) >= 2

            if has_markers or has_quotes:
                attribution_label = "supported_claim"
            else:
                attribution_label = "unsupported_claim"

            return {
                "propaganda_label": propaganda_label,
                "statement_type": statement_type,
                "attribution_label": attribution_label
            }

        except Exception as e:
            logger.error("arabert_classification_error", error=str(e))
            return {
                "propaganda_label": "neutral",
                "statement_type": "fact",
                "attribution_label": "unsupported_claim"
            }


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

        composite = (0.35 * s_auth) + (0.35 * s_neut) + (0.15 * s_attr) + (0.15 * consensus_score)
        return {
            "reliability_score": round(composite, 2),
            "neutrality_score": s_neut,
            "attribution_score": s_attr
        }    


class StoryGrouper:
    def __init__(self) -> None:
        self.device = "cuda" if settings.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_id)
        self.model = AutoModel.from_pretrained(settings.embedding_model_id).to(self.device)
        self.model.eval()
        self.running_mean = np.zeros(768, dtype=np.float32)
        self.total_processed = 0

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_raw_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(768, dtype=np.float32)

        inputs = self.tokenizer(
            [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            
        return embeddings.cpu().squeeze(0).numpy()

    def get_normalized_embedding(self, text: str) -> np.ndarray:
        raw_vector = self.get_raw_embedding(text)
        norm = np.linalg.norm(raw_vector)
        if norm == 0:
            return raw_vector
        return raw_vector / norm

    def update_running_mean(self, raw_vector: np.ndarray) -> None:
        self.total_processed += 1
        self.running_mean += (raw_vector - self.running_mean) / self.total_processed

    def get_centered_normalized_embedding(self, raw_vector: np.ndarray) -> np.ndarray:
        centered = raw_vector - self.running_mean
        norm = np.linalg.norm(centered)
        if norm == 0:
            return centered
        return centered / norm