from __future__ import annotations

import json
from datetime import datetime
from typing import  List, Optional
from db.base import Base
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index,
    Integer, String, Text, JSON, ForeignKey,
)

class ArticleORM(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), unique=True, nullable=False, index=True)
    source_name = Column(String(255), nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    scraped_at = Column(DateTime, default=datetime.utcnow(), nullable=False)
    language = Column(String(10), default="ar")
    reliability_score = Column(Float, nullable=True)
    is_processed = Column(Boolean, default=False, index=True)
    word_count = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_articles_source_published", "source_name", "published_at"),
        Index("idx_articles_processed_scraped", "is_processed", "scraped_at"),
    )