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

    class Config:
        frozen = True


class Settings(BaseSettings):
    app_name: str = "Arabic News Intelligence System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    database_url: str = "sqlite+aiosqlite:///./arabic_news.db"
    database_path: str = "./arabic_news.db"

    fetch_timeout_seconds: int = 15
    max_articles_per_source: int = 50
    article_max_age_hours: int = 48
    fetch_concurrency: int = 8
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()