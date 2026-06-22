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

class ArticleFeedbackSchema(BaseModel):
    article_id:int
    propaganda_prediction: Optional[str] = None
    statement_prediction: Optional[str] = None
    attribution_prediction: Optional[str] = None

    propaganda_correct: Optional[bool] = None
    corrected_propaganda: Optional[str] = None

    statement_correct: Optional[bool] = None
    corrected_statement: Optional[str] = None

    attribution_correct: Optional[bool] = None
    corrected_attribution: Optional[str] = None

    notes: Optional[str] = None


class SummaryFeedbackSchemma(BaseModel):
    query: Optional[str] = None

    generated_summary: str
    user_rating: bool
    feedback_reason: Optional[str] = None
    corrected_summary: Optional[str] = None

    
class FeedbackStatusSchema(BaseModel):
    status: str
    admin_notes: Optional[str] = None

