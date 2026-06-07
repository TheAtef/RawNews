from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class ArticleSchema(BaseModel):
    id: int
    url: str
    source_name: str
    title: str
    content: Optional[str] = None
    published_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None
    bias_score: Optional[float] = None
    bias_direction: Optional[str] = None
    propaganda_score: Optional[float] = None
    reliability_score: Optional[float] = None
    word_count: Optional[int] = None
    persons: List[str] = []
    organizations: List[str] = []
    locations: List[str] = []
    misc: List[str] = []

    class Config:
        from_attributes = True

class SourceInfo(BaseModel):
    name: str
    name_ar: str
    reliability_score: float
    political_lean: str
    region: str
    rss_count: int

class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    total_articles: int
