
from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import  NewsSourceConfig, settings
from core.sources_list import ARABIC_NEWS_SOURCES
from engine.fetcher import ArticleScraper, RSSFetcher
from engine.scraper import RawArticle
from models.orm import ArticleORM

logger = structlog.get_logger(__name__)

class SourceManager:

    def __init__(self) -> None:
        self._sources = ARABIC_NEWS_SOURCES

    async def fetch_all(
        self,
        session: AsyncSession,
        max_per_source: int = 50,
        age_hours: int = 48,
        scrape_full_content: bool = True,
    ) -> List[ArticleORM]:
        cutoff = datetime.utcnow() - timedelta(hours=age_hours)

        rss_fetcher = RSSFetcher(timeout=settings.fetch_timeout_seconds)
        scraper = ArticleScraper(timeout=settings.fetch_timeout_seconds)

        tasks = [
            self._fetch_source(
                source, rss_fetcher, scraper if scrape_full_content else None,
                max_per_source,
            )
            for source in self._sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_raw: List[RawArticle] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "source_fetch_failed",
                    source=self._sources[i].name,
                    error=str(result),
                )
            elif result:
                all_raw.extend(result)

        filtered = [
            a for a in all_raw
            if a.published_at is None or a.published_at >= cutoff
        ]

        logger.info("fetch_complete", total_raw=len(all_raw), after_age_filter=len(filtered))

        new_articles = await self._persist_articles(session, filtered)
        return new_articles

    async def _fetch_source(
        self,
        source: NewsSourceConfig,
        rss_fetcher: RSSFetcher,
        scraper: Optional[ArticleScraper],
        max_per_source: int,
    ) -> List[RawArticle]:
        articles: List[RawArticle] = []

        feed_tasks = [rss_fetcher.fetch_feed(url) for url in source.rss_urls]
        feeds = await asyncio.gather(*feed_tasks, return_exceptions=True)

        for feed in feeds:
            if isinstance(feed, Exception) or feed is None:
                continue
            entries = rss_fetcher.extract_entries(feed, source, max_entries=max_per_source)
            articles.extend(entries)

        if scraper and articles:
            short_articles = [a for a in articles if len(a.content or "") < 300]
            if short_articles:
                scrape_tasks = [scraper.scrape(a.url) for a in short_articles]
                scraped = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                for article, full_content in zip(short_articles, scraped):
                    if isinstance(full_content, str) and len(full_content) > 300:
                        article.content = full_content

        logger.info("source_fetched", source=source.name, count=len(articles))
        return articles

    async def _persist_articles(
        self,
        session: AsyncSession,
        raw_articles: List[RawArticle],
    ) -> List[ArticleORM]:
        if not raw_articles:
            return []

        unique_raw_map = {a.url: a for a in raw_articles}
        urls = list(unique_raw_map.keys())
        
        existing_stmt = select(ArticleORM.url).where(ArticleORM.url.in_(urls))
        existing_result = await session.execute(existing_stmt)
        existing_urls = {row[0] for row in existing_result.fetchall()}

        new_articles: List[ArticleORM] = []
        for url, raw in unique_raw_map.items():
            if url in existing_urls:
                continue
            
            existing_urls.add(url)
            
            article = ArticleORM(
                url=raw.url,
                source_name=raw.source_name,
                title=raw.title,
                content=raw.content,
                published_at=raw.published_at,
                scraped_at=raw.scraped_at,
                reliability_score=raw.reliability_score,
                language=raw.language,
                word_count=len(raw.content.split()) if raw.content else 0,
                is_processed=False,
            )
            session.add(article)
            new_articles.append(article)
            
        try:
            await session.flush()
            logger.info("articles_persisted", new_count=len(new_articles))
        except Exception as e:
            logger.error("persistence_error", error=str(e))
            await session.rollback()
            return []

        return new_articles

    async def search_articles(
        self,
        session: AsyncSession,
        query: str,
        time_window_hours: int = 48,
        limit: int = 200,
    ) -> List[ArticleORM]:

        cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
        tokens = self._preprocessor.tokenize(query, remove_stopwords=True)

        stmt = select(ArticleORM).where(ArticleORM.fetched_at >= cutoff)

        if tokens:
            from sqlalchemy import or_
            conditions = []
            for token in tokens[:5]:
                conditions.extend([
                    ArticleORM.title_clean.contains(token),
                    ArticleORM.content_clean.contains(token),
                ])
            stmt = stmt.where(or_(*conditions))

        stmt = stmt.order_by(ArticleORM.published_at.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    def get_source_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name,
                "name_ar": s.name_ar,
                "reliability_score": s.reliability_score,
                "political_lean": s.political_lean,
                "region": s.region,
                "rss_count": len(s.rss_urls),
            }
            for s in self._sources
        ]