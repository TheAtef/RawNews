from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import newspaper
import cloudscraper
import structlog
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import NewsSourceConfig, settings
from core.sources_list import ARABIC_NEWS_SOURCES
from core.constants import ARABIC_STOPWORDS
from engine.fetcher import ArticleScraper, RSSFetcher, GoogleNews
from engine.scraper import RawArticle
from models.orm import ArticleORM

from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from preprocessing.tokenizer import ArabicTokenizer, ArabicStopwordFilter
from preprocessing.deduplicator import ArticleDeduplicator
from preprocessing.analyzer import HeuristicScorer, StoryGrouper
from preprocessing.utils import is_arabic_text
from preprocessing.ner import NER

logger = structlog.get_logger(__name__)


class SourceManager:
    def __init__(self) -> None:
        self._sources = ARABIC_NEWS_SOURCES
        self._cleaner = ArabicNewsCleaner(remove_numbers=False, keep_quotes=True)
        self._normalizer = ArabicNormalizer()
        self._tokenizer = ArabicTokenizer()
        self._stop_filter = ArabicStopwordFilter(extra_stopwords=ARABIC_STOPWORDS)
        self._scorer = HeuristicScorer()
        self._ner=NER()

        
        self._deduplicator = ArticleDeduplicator(similarity_threshold=0.82)
        self._grouper = StoryGrouper(overlap_threshold=0.18)

        self._warmed_up = False

    async def _warm_up_processors(self, session: AsyncSession) -> None:
        if self._warmed_up:
            return

        cutoff = datetime.utcnow() - timedelta(hours=72)
        stmt = (
            select(ArticleORM)
            .where(ArticleORM.scraped_at >= cutoff)
            .order_by(ArticleORM.id.asc())
        )
        result = await session.execute(stmt)
        recent_articles = result.scalars().all()

        for art in recent_articles:
            if art.content:
                self._deduplicator.seed_known_article(str(art.id), art.content)
            elif art.title:
                self._deduplicator.seed_known_article(str(art.id), art.title)

            if art.cluster_id and art.content_clean:
                tokens = self._tokenizer.tokenize(art.content_clean)
                filtered = self._stop_filter.filter(tokens)
                self._grouper.seed_cluster_anchor(art.cluster_id, filtered)

        self._warmed_up = True
        logger.info("processors_warmed_up", loaded_count=len(recent_articles))

    async def fetch_all(
        self,
        session: AsyncSession,
        max_per_source: int = 50,
        age_hours: int = 48,
        scrape_full_content: bool = True,
    ) -> List[ArticleORM]:
        await self._warm_up_processors(session)
        
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
            
            if not is_arabic_text(raw.title + " " + (raw.content or "")):
                continue

            cleaned_title = self._cleaner.clean(raw.title)
            normalized_title = self._normalizer.normalize(cleaned_title)

            cleaned_content = self._cleaner.clean(raw.content or "")
            normalized_content = self._normalizer.normalize(cleaned_content)
            entities=self._ner.extract_entities(cleaned_content)

            tokens = self._tokenizer.tokenize(normalized_content)
            filtered_tokens = self._stop_filter.filter(tokens)

            is_dup, _ = self._deduplicator.is_duplicate(raw.url, normalized_content)
            if is_dup:
                continue
            
            existing_urls.add(url)

            temp_article_payload = {
                "url": raw.url,
                "title": raw.title,
                "tokens": filtered_tokens
            }
            assigned_cluster_id = self._grouper.add_to_cluster(temp_article_payload, filtered_tokens)

            scores = self._scorer.evaluate_article(
                source_name=raw.source_name,
                raw_text=raw.content or "",
                tokens=filtered_tokens
            )
            article_orm = ArticleORM(
                url=raw.url,
                source_name=raw.source_name,
                title=raw.title,
                content=raw.content,
                title_clean=normalized_title,
                content_clean=normalized_content,

                persons=entities["person"],
                organizations=entities["organization"],
                locations=entities["location"],
                misc=entities["misc"],

                
                cluster_id=assigned_cluster_id,
                reliability_score=scores["reliability_score"],
                neutrality_score=scores["neutrality_score"],
                attribution_score=scores["attribution_score"],
                published_at=raw.published_at,
                scraped_at=raw.scraped_at,
                language=raw.language,
                word_count=len(raw.content.split()) if raw.content else 0,
                is_processed=True,
            )
            session.add(article_orm)
            new_articles.append(article_orm)
            
        try:
            await session.flush()
            logger.info("articles_persisted", new_count=len(new_articles))
        except Exception as e:
            logger.error("persistence_error", error=str(e))
            await session.rollback()
            return []

        return new_articles

    def preprocess_query(self, query: str) -> List[str]:
        cleaned = self._cleaner.clean(query)
        normalized = self._normalizer.normalize(cleaned)
        tokens = self._tokenizer.tokenize(normalized)
        return self._stop_filter.filter(tokens)

    async def search_articles(
            self,
            session: AsyncSession,
            query: str,
            time_window_hours: int = 48,
            limit: int = 200,
        ) -> List[ArticleORM]:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            tokens = self.preprocess_query(query)

            stmt = select(ArticleORM).where(ArticleORM.published_at >= cutoff)

            if tokens:
                token_conditions = []                
                for token in tokens[:5]:
                    token_conditions.append(
                        or_(
                            ArticleORM.title_clean.contains(token),
                            ArticleORM.content_clean.contains(token),
                        )
                    )
                
                stmt = stmt.where(and_(*token_conditions))

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
        
    async def search_google_news(
        self, 
        query: str, 
        time_window: str = "7d", 
        limit: int = 20, 
        scrape_full_content: bool = True,
        session: Optional[AsyncSession] = None) -> List[RawArticle]:
        google_news = GoogleNews(query=query, time_window = time_window, limit=limit, scrape_full_content=scrape_full_content)
        entries = await google_news.fetch_news()
        scraper = ArticleScraper(timeout=settings.fetch_timeout_seconds)
        articles: List[RawArticle] = []
        for entry in entries:
            try:
                text = ""
                data = newspaper.article(entry.link)
                if hasattr(data, 'text_cleaned') and data.text_cleaned != "":
                    text = data.text_cleaned
                elif hasattr(data, 'text') and data.text != "":
                    text = data.text
                else: 
                    text = await scraper.scrape(entry.link)
                    
                article = RawArticle(
                    url=entry.link,
                    title=entry.title.rsplit(' - ', 1)[0],
                    content=text,
                    source_name=entry.source.title,
                    published_at= datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S GMT"),
                    scraped_at= datetime.utcnow(),
                    reliability_score=None,
                    language='ar',
                )
                articles.append(article)
                logger.info("google_news_article_fetched", title=entry.title, url=entry.link)
            except Exception as e:
                logger.error("google_news_article_error", url=entry.link, error=str(e))
                continue
        if session:
            await self._persist_articles(session=session, raw_articles=articles)
        return articles