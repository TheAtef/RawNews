"""
main.py — FastAPI application.

Endpoints:
    GET  /health         — System health check
    GET  /sources        — List configured news sources
    GET  /articles       — List recent articles 
    GET  /article/{id}   — Retrieve article by ID
    GET  /stats          — Pipeline statistics


Background tasks:
    - Periodic news fetching every 30 minutes
"""

from __future__ import annotations

import collections
import collections.abc
from unittest import result

from engine.synthesizer import NewsSynthesizer

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal
from run import run_intelligence_pipeline

import structlog
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text,or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from fastapi import BackgroundTasks
from services.background_fetch import (background_fetch_loop,stop_background_fetch,)

from core.config import settings
from db.session import get_db, init_db
from engine.manager import SourceManager

from models.orm import ArticleORM,ArticleFeedbackORM,SummaryFeedbackORM,FeedbackStatus,ClusterORM
from models.schemas import ArticleSchema, HealthResponse, SourceInfo,ArticleFeedbackSchema,SummaryFeedbackSchemma,FeedbackStatusSchema,ArticleDetailsSchema,ArticleCardSchema,ArticleListResponse,ArticleSourceSchema,ArticleSourcesResponse,SearchResponseSchema
from utils.logging import configure_logging
from services.training_service import start_retraining_if_needed
from services.summary_training_service import start_summary_retraining_if_needed
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)



source_manager = SourceManager()
news_synthesizer = NewsSynthesizer() 
# _background_fetch_running = False



# async def background_fetch_loop(interval_minutes: int = 30) -> None:
#     """Periodically fetch fresh news from all sources."""
#     global _background_fetch_running
#     _background_fetch_running = False

#     # await asyncio.sleep(5)

#     while _background_fetch_running:
#         try:
#             logger.info("background_fetch_starting")
#             from db.session import AsyncSessionLocal
#             async with AsyncSessionLocal() as db:
#                 articles = await source_manager.fetch_all(
#                     session=db,
#                     max_per_source=settings.max_articles_per_source,
#                     age_hours=settings.article_max_age_hours,
#                     scrape_full_content=True,
#                 )
#                 await db.commit()
#                 logger.info("background_fetch_done", new_articles=len(articles))
#         except Exception as e:
#             logger.error("background_fetch_error", error=str(e))

#         await asyncio.sleep(interval_minutes * 60)



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("startup_begin", version=settings.app_version)
    await init_db()
    logger.info("database_initialized")
    # fetch_task = asyncio.create_task(background_fetch_loop(interval_minutes=30))
    # logger.info("background_fetch_scheduled")

    # logger.info("startup_complete", host=settings.api_host, port=settings.api_port)

    yield 
    # stop_background_fetch()

    # fetch_task.cancel()
    # try:
    #     await fetch_task
    # except asyncio.CancelledError:
    #     pass
    # logger.info("shutdown_complete")
    
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Arabic News Intelligence System — "
        "Bias detection, event clustering, contradiction analysis, "
        "and neutral evidence-based summarization."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(404)
async def not_found_handler(request: Any, exc: Any) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "Resource not found", "path": str(request.url)})


@app.exception_handler(500)
async def server_error_handler(request: Any, exc: Any) -> JSONResponse:
    logger.error("unhandled_error", path=str(request.url), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Check logs for details."},
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    db_ok = False
    total_articles = 0

    try:
        result = await db.execute(select(func.count(ArticleORM.id)))
        total_articles = result.scalar() or 0
        db_ok = True
    except Exception as e:
        logger.error("health_db_error", error=str(e))

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=settings.app_version,
        db_connected=db_ok,
        total_articles=total_articles,
    )


@app.get(
    "/sources",
    response_model=List[SourceInfo],
    summary="List news sources",
    tags=["Sources"],
)
async def list_sources() -> List[SourceInfo]:
    return [SourceInfo(**s) for s in source_manager.get_source_info()]


@app.get(
    "/article/{article_id}",
    response_model=ArticleSchema,
    summary="Get article by ID",
    tags=["Articles"],
)
async def get_article(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleSchema:
    result = await db.execute(
        select(ArticleORM).where(ArticleORM.id == article_id)
    )
    article = result.scalar_one_or_none()
    if article is None:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
    return ArticleSchema.model_validate(article)


@app.get(
    "/articles",
    response_model=List[ArticleSchema],
    summary="List recent articles",
    tags=["Articles"],
)
async def list_articles(
    source: Optional[str] = Query(default=None, description="Filter by source name"),
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[ArticleSchema]:
    stmt = select(ArticleORM).order_by(ArticleORM.id.desc())
    if source:
        stmt = stmt.where(ArticleORM.source_name == source)
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    articles = result.scalars().all()
    return [ArticleSchema.model_validate(a) for a in articles]


@app.get(
    "/articles/feed",
    summary="News feed",
    tags=["Articles"],
)
async def get_news_feed(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    source: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        stmt = select(ArticleORM)

        if source:
            stmt = stmt.where(
                ArticleORM.source_name == source
            )

        total = await db.scalar(
            select(func.count()).select_from(stmt.subquery())
        )

        result = await db.execute(
            stmt.order_by(ArticleORM.published_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        articles = result.scalars().all()

        if not articles:
            return {
                "page": page,
                "page_size": page_size,
                "total": total,
                "has_next": False,
                "articles": [],
                "message": (
                    "No articles found."
                    if page == 1
                    else "No articles found for this page."
                ),
            }

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": page * page_size < total,
            "articles": [
                ArticleCardSchema.model_validate(article)
                for article in articles
            ],
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve articles."
        )

import time
@app.get(
    "/search",
    response_model=SearchResponseSchema,
    summary="Search, analyze, and summarize news articles",
    tags=["Search"],
)
async def search_news(
    query: str = Query(..., min_length=2),
    time_window: Literal["1h", "1d", "3d", "7d", "30d"] = Query(default="3d"),
    limit: int = Query(default=10, ge=1, le=50),
):
    start = time.perf_counter()

    limit = min(limit, 10)
    result = await run_intelligence_pipeline(
        search_query=query,
        time_window=time_window,
        limit=limit,manager=source_manager,
        synthesizer=news_synthesizer
    )
    elapsed = time.perf_counter() - start

    logger.info(
        "search_completed",
        elapsed_seconds=round(elapsed, 2)
    )

    if result is None:
        return SearchResponseSchema(
            status="no_results",
            query=query,
            time_window=time_window,
            clusters=[]
        )

    return SearchResponseSchema(
        status="success",
        **result
    )
@app.get(
    "/articles/{article_id}/sources",
    response_model=ArticleSourcesResponse,
    summary="Get all news sources covering the same event",
    tags=["Articles"],
)
async def get_article_sources(
    article_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        article = await db.get(ArticleORM, article_id)

        if article is None:
            raise HTTPException(
                status_code=404,
                detail="Article not found."
            )

        if article.cluster_id is None:
            return ArticleSourcesResponse(
                total=0,
                message="This article does not belong to any cluster.",
                sources=[]
            )

        stmt = (
            select(
                ArticleORM.source_name,
                func.count(ArticleORM.id).label("articles_count")
            )
            .where(
                ArticleORM.cluster_id == article.cluster_id
            )
            .group_by(ArticleORM.source_name)
            .order_by(
                func.count(ArticleORM.id).desc(),
                ArticleORM.source_name
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        if not rows:
            return ArticleSourcesResponse(
                total=0,
                message="No sources found.",
                sources=[]
            )

        return ArticleSourcesResponse(
            total=len(rows),
            sources=[
                ArticleSourceSchema(
                    name=row.source_name,
                    articles_count=row.articles_count,
                )
                for row in rows
            ]
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve article sources."
        )


@app.get(
    "/articles/{article_id}",
    response_model=ArticleDetailsSchema,
    summary="Get article details",
    tags=["Articles"],
)
async def get_article_details(
    article_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await db.execute(
            select(ArticleORM).where(
                ArticleORM.id == article_id
            )
        )

        article = result.scalar_one_or_none()

        if article is None:
            raise HTTPException(
                status_code=404,
                detail="Article not found."
            )

        return ArticleDetailsSchema.model_validate(article)

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve article."
        )






@app.get(
    "/articles/{article_id}/related",
    response_model=ArticleListResponse,
    summary="Get related articles from the same event",
    tags=["Articles"],
)
async def get_related_articles(
    article_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    try:
        article = await db.get(ArticleORM, article_id)

        if article is None:
            raise HTTPException(
                status_code=404,
                detail="Article not found."
            )

        if article.cluster_id is None:
            return ArticleListResponse(
                page=page,
                page_size=page_size,
                total=0,
                has_next=False,
                message="This article does not belong to any cluster.",
                articles=[]
            )

        stmt = (
            select(ArticleORM)
            .where(
                ArticleORM.cluster_id == article.cluster_id,
                ArticleORM.id != article.id
            )
            .order_by(ArticleORM.verified.desc(),
                ArticleORM.reliability_score.desc(),
                ArticleORM.published_at.desc())
                    )

        total = await db.scalar(
            select(func.count()).select_from(stmt.subquery())
        )

        result = await db.execute(
            stmt.offset((page - 1) * page_size)
            .limit(page_size)
        )

        related_articles = result.scalars().all()

        if not related_articles:
            return ArticleListResponse(
                page=page,
                page_size=page_size,
                total=0,
                has_next=False,
                message="No related articles found.",
                articles=[]
            )

        return ArticleListResponse(
            page=page,
            page_size=page_size,
            total=total,
            has_next=(page * page_size) < total,
            articles=[
                ArticleCardSchema.model_validate(article)
                for article in related_articles
            ]
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve related articles."
        )


    





@app.get(
    "/stats",
    summary="Pipeline statistics",
    tags=["System"],
)
async def get_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        total_articles = (await db.execute(select(func.count(ArticleORM.id)))).scalar() or 0
        processed_articles = (
            await db.execute(
                select(func.count(ArticleORM.id)).where(ArticleORM.is_processed == True)
            )
        ).scalar() or 0

        # Per-source count
        source_counts_result = await db.execute(
            select(ArticleORM.source_name, func.count(ArticleORM.id).label("count"))
            .group_by(ArticleORM.source_name)
            .order_by(func.count(ArticleORM.id).desc())
        )
        source_counts = {row[0]: row[1] for row in source_counts_result.fetchall()}

    except Exception as e:
        logger.error("stats_db_error", error=str(e))
        total_articles = processed_articles  = 0
        source_counts = {}

    return {
        "articles": {
            "total": total_articles,
            "processed": processed_articles,
            "unprocessed": total_articles - processed_articles,
            "by_source": source_counts,
        },
    }

######################################################################3
# @app.post(
#     "/fetch",
#     summary="Trigger manual news fetch",
#     tags=["Sources"],
# )
# async def trigger_fetch(
#     background_tasks: BackgroundTasks,
#     db: AsyncSession = Depends(get_db),
#     max_per_source: Optional[int] = Query(default=50, ge=1, description="Maximum articles per source"),
#     age_hours: int = Query(default=48, ge=1),
#     scrape_full_content: bool = Query(default=True),
# ) -> Dict[str, Any]:
#     async def _do_fetch() -> None:
#         from db.session import AsyncSessionLocal
#         async with AsyncSessionLocal() as session:
#             try:
#                 articles = await source_manager.fetch_all(
#                     session=session,
#                     max_per_source=max_per_source,
#                     age_hours=age_hours,
#                     scrape_full_content=scrape_full_content
#                 )
#                 await session.commit()
#                 logger.info("manual_fetch_done", count=len(articles))
#             except Exception as e:
#                 logger.error("manual_fetch_error", error=str(e))

#     background_tasks.add_task(_do_fetch)
#     return {"status": "fetch_scheduled", "message": "News fetch started in background"}
#################################################################################################






@app.post(
    "/article-feedback"
)
async def save_article_feedback(
    feedback:ArticleFeedbackSchema,
    db:AsyncSession=Depends(get_db)
):
    article=await db.get(ArticleORM,feedback.article_id)
    if article is None:
        raise HTTPException(status_code=404,detail="article not found")


    item=ArticleFeedbackORM(
        article_id=feedback.article_id,

        propaganda_prediction=article.propaganda_label,
        statement_prediction=article.statement_type,
        attribution_prediction=article.attribution_label,

        propaganda_correct=feedback.propaganda_correct,
        corrected_propaganda=feedback.corrected_propaganda,

        statement_correct=feedback.statement_correct,
        corrected_statement=feedback.corrected_statement,

        attribution_correct=feedback.attribution_correct,
        corrected_attribution=feedback.corrected_attribution,

        notes=feedback.notes
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    return{
        "status":"saved",
        "feedback_id":item.id
    }


@app.post("/summary_feedback")
async def save_summary_feedback(
    feedback:SummaryFeedbackSchemma,
    db:AsyncSession=Depends(get_db)
):
    cluster = await db.get(ClusterORM,feedback.cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail="cluster not found"
        )
    
    item = SummaryFeedbackORM(
        cluster_id=cluster.id,
        query=feedback.query,
        user_rating=feedback.user_rating,
        feedback_reason=feedback.feedback_reason,
        generated_summary=feedback.generated_summary,
        corrected_summary=feedback.corrected_summary
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    return{
        "status":"saved",
        "feedback_id":item.id
    }
@app.patch("/article-feedback/{feedback_id}/status")
async def article_feedback_status(
    feedback_id: int,
    status:FeedbackStatusSchema ,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ArticleFeedbackORM)
        .where(ArticleFeedbackORM.id == feedback_id)
    )
    
    feedback = result.scalar_one_or_none()

    if feedback is None:
        raise HTTPException(
            status_code=404,
            detail="Feedback not found"
        )

    feedback.status = status.status
    feedback.admin_notes = status.admin_notes

    await db.commit()
    await db.refresh(feedback)


    if feedback.status == FeedbackStatus.APPROVED:
        background_tasks.add_task(start_retraining_if_needed)

    return {
        "message": "Feedback updated successfully",
        "status": feedback.status,
    }


@app.patch("/summary-feedback/{feedback_id}/status")
async def summary_feedback_status(
    feedback_id: int,
    status: FeedbackStatusSchema,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(SummaryFeedbackORM)
        .where(SummaryFeedbackORM.id == feedback_id)
    )

    feedback = result.scalar_one_or_none()

    if feedback is None:
        raise HTTPException(
            status_code=404,
            detail="Feedback not found"
        )

    feedback.status = status.status
    feedback.admin_notes = status.admin_notes

    await db.commit()
    await db.refresh(feedback)

    if feedback.status == FeedbackStatus.APPROVED:
        background_tasks.add_task(start_summary_retraining_if_needed)
    

    return {
        "message": "Feedback updated successfully",
        "status": feedback.status
    }
@app.get("/article-feedback/pending")
async def get_pending_feedback(
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ArticleFeedbackORM,ArticleORM)
        .join(ArticleORM,ArticleORM.id == ArticleFeedbackORM.article_id)
        .where(ArticleFeedbackORM.status == FeedbackStatus.PENDING)
        .order_by(ArticleFeedbackORM.id.desc())
    )

    feedbacks = result.all()
    if not feedbacks:
        return {
            "message": "No pending feedback found.",
            "data": []
        }

    return [
        {
            "id": feedback.id,
            "article_id": article.id,
            "article_title": article.title,
            "submitted_at": feedback.created_at,
            "propaganda_correct": feedback.propaganda_correct,
            "statement_correct": feedback.statement_correct,
            "attribution_correct": feedback.attribution_correct,
        }
        for feedback ,article in feedbacks
    ]
@app.get("/article-feedback/{feedback_id}")
async def get_feedback_details(
    feedback_id: int,
    db: AsyncSession = Depends(get_db)
):
    stmt = (
        select(ArticleFeedbackORM, ArticleORM).join(ArticleORM,ArticleORM.id == ArticleFeedbackORM.article_id)
        .where(ArticleFeedbackORM.id == feedback_id))
    result = await db.execute(stmt)
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404,detail="Feedback not found.")
    feedback, article = row
    return {
        "message": "Feedback details retrieved successfully.",
        "data": {
            "article": {
                "id": article.id,
                "title": article.title,
                "content": article.content,
                "neutrality_score": article.neutrality_score,
                "reliability_score": article.reliability_score,

                "propaganda_prediction": article.propaganda_label,
                "statement_prediction": article.statement_type,
                "attribution_prediction": article.attribution_label,
            },

            "feedback": {
                "id": feedback.id,
                "status": feedback.status,

                "propaganda_correct": feedback.propaganda_correct,
                "corrected_propaganda": feedback.corrected_propaganda,

                "statement_correct": feedback.statement_correct,
                "corrected_statement": feedback.corrected_statement,

                "attribution_correct": feedback.attribution_correct,
                "corrected_attribution": feedback.corrected_attribution,

                "notes": feedback.notes,
            }
        }
    }


@app.get("/clusters/{cluster_id}")
async def get_cluster(
    cluster_id: int,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ClusterORM)
        .options(selectinload(ClusterORM.articles))
        .where(ClusterORM.id == cluster_id)
    )

    cluster = result.scalar_one_or_none()

    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail="Cluster not found"
        )

    return {
        "cluster_id": cluster.id,
        "summary": cluster.summary,
        "articles": [
            {
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "source_name": article.source_name,
                "reliability_score": article.reliability_score,
                "neutrality_score": article.neutrality_score,
                "attribution_score":article.attribution_score,
                "verified": article.verified,
                "statement_type": article.statement_type,     
                "attribution_label": article.attribution_label,
                "propaganda_label": article.propaganda_label, 
                "published_at": article.published_at,
            }
            for article in cluster.articles
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,        
        reload=settings.debug,
        log_config=None,   
        access_log=False,
    )
