from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class NewsSourceConfig(BaseModel):
    name: str
    name_ar: str
    rss_urls: List[str]
    base_url: str
    reliability_score: float = Field(ge=0.0, le=1.0)
    political_lean: str = "unknown"  
    region: str = "unknown"
    use_local_gemma_pipeline: bool = False
    class Config:
        frozen = True


class Settings(BaseSettings):
    app_name: str = "Arabic News Intelligence System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    database_url: str 
    # database_path: str = "./arabic_news.db"

    fetch_timeout_seconds: int = 7
    max_articles_per_source: int = 50
    article_max_age_hours: int = 48
    fetch_concurrency: int = 8
    blocked_websites: List[str] = ["www.syriahr.com", "www.reuters.com"]
    
    device: str = "cuda"  
    
    # arabert configurations for analysis and clustering
    embedding_model_id: str = "aubmindlab/bert-base-arabertv02"
    sentiment_model_id: str = "./train/models/fine_tuned_arabert_statement_type"
    multi_sentiment_model_id: str = "./train/train/models/fine_tuned_qwen_propaganda"

    # gemma 2 configuration strictly for summarization
    # gemma_model_id: str = "./train/models/gemma2_arabic_summarizer_adapter"
    # gemma_model_id: str = "google/gemma-3-1b-it"
    gemma_model_id: str = "./train/models/gemma3_1b_arabic_summarizer_adapter"
    # gemma_api_url: str = "http://localhost:11434/v1" 
    # gemma_api_key: str = "ollama"
    use_local_gemma_pipeline: bool = False  
    
    # clustering Parameters
    clustering_similarity_threshold: float = 0.78
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()