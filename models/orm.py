from __future__ import annotations
from datetime import datetime, timezone 
from db.base import Base
from enum import Enum as pyenum
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index,
    Integer, String, Text, JSON,ForeignKey,Enum
)
from sqlalchemy.orm import relationship

class ArticleORM(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), unique=True, nullable=False, index=True)
    source_name = Column(String(255), nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    
    title_clean = Column(Text, nullable=True)
    content_clean = Column(Text, nullable=True)
    
    persons = Column(JSON, nullable=True)
    organizations = Column(JSON, nullable=True)
    locations = Column(JSON, nullable=True)
    misc = Column(JSON, nullable=True)
    
    cluster_id = Column(Integer, nullable=True, index=True)
    reliability_score = Column(Float, nullable=True)
    neutrality_score = Column(Float, nullable=True)
    attribution_score = Column(Float, nullable=True)
    
    propaganda_label = Column(String(50), nullable=True)      
    statement_type = Column(String(50), nullable=True)         
    attribution_label = Column(String(100), nullable=True)      
    verified = Column(Boolean, default=False, nullable=False)

    published_at = Column(DateTime(timezone=True), nullable=True, index=True)
    scraped_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    language = Column(String(10), default="ar")
    is_processed = Column(Boolean, default=False, index=True)
    word_count = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_articles_source_published", "source_name", "published_at"),
        Index("idx_articles_processed_scraped", "is_processed", "scraped_at"),
    )
    feedbacks=relationship("ArticleFeedbackORM",back_populates="article")

class FeedbackStatus(pyenum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ArticleFeedbackORM(Base):
    __tablename__ = "article_feedback"
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=True)


    propaganda_prediction = Column(String(50),nullable=True)
    statement_prediction = Column(String(50),nullable=True)
    attribution_prediction = Column(String(100),nullable=True)

    propaganda_correct = Column(Boolean,nullable=True)
    corrected_propaganda = Column(String(50),nullable=True)

    statement_correct = Column(Boolean,nullable=True)
    corrected_statement = Column(String(50),nullable=True)

    attribution_correct = Column(Boolean, nullable=True)
    corrected_attribution = Column(String(100), nullable=True)

    notes=Column(String,nullable=True)

    article = relationship("ArticleORM",back_populates="feedbacks")
    training_job_id = Column(Integer,ForeignKey("training_job.id"),nullable=True)
    training_job = relationship("TrainingJobORM",back_populates="feedbacks")

    status = Column(
        Enum(FeedbackStatus,values_callable=lambda enum: [e.value for e in enum]),default=FeedbackStatus.PENDING,nullable=False
    )
    admin_notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SummaryFeedbackORM(Base):
    __tablename__="summary_feedback"
    id=Column(Integer,primary_key=True)
    cluster_id = Column(Integer, nullable=False)

    generated_summary = Column(Text, nullable=False)

    corrected_summary = Column(Text, nullable=True)
    query=Column(String(255),nullable=True)
    user_rating=Column(Boolean,nullable=False)
    feedback_reason=Column(String(100),nullable=True)
    created_at=Column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))
    status = Column(
        Enum(FeedbackStatus, values_callable=lambda enum: [e.value for e in enum]),
        default=FeedbackStatus.PENDING,
        nullable=False
    )
    admin_notes = Column(Text, nullable=True)
class TrainingJobORM(Base):
    __tablename__="training_job"
    id=Column(Integer,primary_key=True)
    feedback_count=Column(Integer,nullable=False)
    accuracy=Column(Float,nullable=True)
    status=Column(Enum("PENDING","RUNNING","COMPLETED","FAILED",name="training_job_status"),default="PENDING",nullable=False)
    statement_accuracy = Column(Float, nullable=True)
    propaganda_accuracy = Column(Float, nullable=True)
    propaganda_f1 = Column(Float, nullable=True)
    parse_failure_rate = Column(Float, nullable=True)
    model_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    started_at = Column(DateTime(timezone=True),nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    error_message = Column(Text, nullable=True)
    feedbacks = relationship("ArticleFeedbackORM", back_populates="training_job")