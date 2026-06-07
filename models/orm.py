from __future__ import annotations
from datetime import datetime
from db.base import Base
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index,
    Integer, String, Text,JSON
)

class ArticleORM(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), unique=True, nullable=False, index=True)
    source_name = Column(String(255), nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    
    title_clean = Column(Text, nullable=True)
    content_clean = Column(Text, nullable=True)
    
    persons=Column(JSON, nullable=True)
    organizations=Column(JSON, nullable=True)
    locations=Column(JSON, nullable=True)
    misc=Column(JSON, nullable=True)
    
    cluster_id = Column(Integer, nullable=True, index=True)
    reliability_score = Column(Float, nullable=True)
    neutrality_score = Column(Float, nullable=True)
    attribution_score = Column(Float, nullable=True)

    
    published_at = Column(DateTime, nullable=True, index=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    language = Column(String(10), default="ar")
    is_processed = Column(Boolean, default=False, index=True)
    word_count = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_articles_source_published", "source_name", "published_at"),
        Index("idx_articles_processed_scraped", "is_processed", "scraped_at"),
    )