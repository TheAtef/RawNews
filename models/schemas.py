from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from .orm import FeedbackStatus

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

class ArticleFeedDetailsSchema(BaseModel):
    id: int
    source_name: str
    title: str
    published_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None
    bias_score: Optional[float] = None
    bias_direction: Optional[str] = None
    reliability_score: Optional[float] = None
    neutrality_score: Optional[float] = None
    attribution_score:Optional[float] = None
    statement_type: Optional[str] = None
    attribution_label: Optional[str] = None
    propaganda_label: Optional[str] = None
    verified: bool

    word_count: Optional[int] = None
    persons: List[str] = []
    organizations: List[str] = []
    locations: List[str] = []
    misc: List[str] = []

    class Config:
        from_attributes = True
class ArticleCardSchema(BaseModel):
    id: int
    title: str
    source_name: str

    published_at: Optional[datetime] = None

    reliability_score: Optional[float] = None
    neutrality_score: Optional[float] = None

    verified: bool
    class Config:
        from_attributes = True

class ArticleDetailsSchema(BaseModel):
    id: int
    url: str

    source_name: str

    title: str
    content: Optional[str] = None

    published_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None

    reliability_score: Optional[float] = None
    neutrality_score: Optional[float] = None
    attribution_score: Optional[float] = None

    propaganda_label: Optional[str] = None
    statement_type: Optional[str] = None
    attribution_label: Optional[str] = None

    verified: bool

    cluster_id: Optional[int] = None

    language: Optional[str] = None
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
    propaganda_correct: Optional[bool] = None
    corrected_propaganda: Optional[str] = None

    statement_correct: Optional[bool] = None
    corrected_statement: Optional[str] = None

    attribution_correct: Optional[bool] = None
    corrected_attribution: Optional[str] = None

    notes: Optional[str] = None


class SummaryFeedbackSchemma(BaseModel):
    cluster_id: int
    query: Optional[str] = None
    generated_summary: str
    user_rating: bool
    feedback_reason: Optional[str] = None
    corrected_summary: Optional[str] = None

class FeedbackStatusSchema(BaseModel):
    status: FeedbackStatus
    admin_notes: Optional[str] = None

class ArticleListResponse(BaseModel):
    page: int
    page_size: int
    total: int
    has_next: bool
    message: Optional[str] = None
    articles: List[ArticleCardSchema]



class ArticleSourceSchema(BaseModel):
    name: str
    articles_count: int
    


class ArticleSourcesResponse(BaseModel):
    total: int
    message: Optional[str] = None
    sources: List[ArticleSourceSchema]


class ArticleSearchResultSchema(BaseModel):
    id: int
    url:str

    title: str
    source_name: str

    published_at: Optional[datetime] = None

    reliability_score: Optional[float] = None
    neutrality_score: Optional[float] = None

    verified: bool

    statement_type: Optional[str] = None
    attribution_label: Optional[str] = None
    propaganda_label: Optional[str] = None

    cluster_id: Optional[int] = None

    class Config:
        from_attributes = True


class SearchClusterSchema(BaseModel):
    cluster_id: int

    summary: Optional[str] = None

    articles: List[ArticleSearchResultSchema]

class SearchResponseSchema(BaseModel):
    status: str

    query: str
    time_window: str

    clusters: List[SearchClusterSchema]

