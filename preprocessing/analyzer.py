from __future__ import annotations
import re
import torch
import numpy as np
import structlog
from typing import List, Dict, Set, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from core.config import settings
from preprocessing.propaganda_features import (LOADED_PHRASES, calculate_loaded_words_ratio)
# from core.constants import BIAS_INDICATORS
from core.sources_list import ARABIC_NEWS_SOURCES
from core.constants import ATTRIBUTION_MARKERS, OPINION_MARKERS

logger = structlog.get_logger(__name__)

SOURCE_AUTHORITY: Dict[str, float] = {
    source.name: source.reliability_score for source in ARABIC_NEWS_SOURCES
}

# LOADED_WORDS: Set[str] = set()
# for category_dict in BIAS_INDICATORS.values():
#     LOADED_WORDS.update(category_dict.keys())


# preprocessing/analyzer.py
from train.prompt_utils import build_messages, parse_output
from transformers import AutoModelForCausalLM 

class AraBERTClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.multi_sentiment_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.multi_sentiment_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.enabled = True
        except Exception as e:
            logger.error("qwen_classifier_init_failed", error=str(e))
            self.enabled = False

    def classify_propaganda(self, text: str, title: str, loaded_words_ratio: float) -> str:
        if not self.enabled or not text.strip():
            logger.warning("classifier_disabled_or_empty_text", enabled=self.enabled)
            return "neutral"

        try:
            messages = build_messages(
                title=title,
                content=text,
                loaded_words_ratio=loaded_words_ratio,
                include_answer=False
            )
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.3,  
                    do_sample=True,   
                    top_p=0.85,       
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            prediction = parse_output(decoded_output)

            print("\n================ [PROPAGANDA MODEL DEBUG] ================")
            print(f" Title: {title[:60]}...")
            print(f" Loaded Words Ratio: {loaded_words_ratio * 100:.2f}%")
            print(f" Raw Model Text Output: '{decoded_output}'")
            print(f" Final Parsed Label: '{prediction}'")
            print("=========================================================\n")

            return prediction

        except Exception as e:
            logger.error("qwen_inference_error", error=str(e))
            return "Error"
class HeuristicScorer:
    def __init__(self, source_registry: Dict[str, float] = None):
        self.source_registry = source_registry if source_registry is not None else SOURCE_AUTHORITY

    def calculate_source_authority(self, source_name: str) -> float:
        return self.source_registry.get(source_name, 0.40)

    def calculate_neutrality_score(self, tokens: List[str]) -> float:
        density = calculate_loaded_words_ratio(tokens)
        score = max(0.0, 1.0 - (density * 15.0))
        return round(score, 2)

    def calculate_attribution_score(self, text: str, title: str = "") -> float:
        score = 0.0
        combined_text = (title + " " + text)
        
        quotes = re.findall(r'["«»“”]', combined_text)
        if len(quotes) >= 2:
            score += 0.40            
        
        if title and re.search(r'\w+\s*:\s*\w+', title):
            score += 0.35

        marker_matches = sum(1 for marker in ATTRIBUTION_MARKERS if marker in combined_text)
        score += min(0.60, marker_matches * 0.15)
        return round(score, 2)

    def determine_attribution_label(self, attribution_score: float) -> str:
        return "supported_claim" if attribution_score >= 0.30 else "unsupported_claim"

    def determine_statement_type(self, title: str, raw_text: str, neutrality_score: float) -> str:
        combined = (title + " " + raw_text).lower()
        
        if any(marker in combined for marker in OPINION_MARKERS):
            return "opinion"
        
        if neutrality_score < 0.50:
            return "opinion"
            
        return "reporting"

    def evaluate_article(
        self, 
        source_name: str, 
        raw_text: str, 
        tokens: List[str], 
        title: str = "",
        consensus_score: float = 1.0
    ) -> Dict[str, float]:
        s_auth = self.calculate_source_authority(source_name)
        s_neut = self.calculate_neutrality_score(tokens)
        s_attr = self.calculate_attribution_score(raw_text, title=title) # Pass title down

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