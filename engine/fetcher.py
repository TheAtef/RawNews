import asyncio
from datetime import datetime
import re
from typing import Any

from bs4 import BeautifulSoup
import feedparser
from typing import List, Optional
import structlog

from core.config import NewsSourceConfig
from engine.scraper import RawArticle
logger = structlog.get_logger(__name__)

import cloudscraper


class RSSFetcher:


    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout
        self._cs = cloudscraper.create_scraper()

    async def fetch_feed(self, url: str) -> Optional[feedparser.FeedParserDict]:
        try:
            with self._cs.get(url, timeout=self._timeout) as response:
                if response.status_code != 200:
                    logger.warning("rss_fetch_error", url=url, status=response.status_code)
                    return None
                text = response.content
                feed = feedparser.parse(text)
                if feed.bozo and not feed.entries:
                    logger.warning("rss_parse_error", url=url, error=str(feed.bozo_exception))
                    return None
                logger.info("rss_fetched", url=url, entries=len(feed.entries))
                return feed
        except Exception as e:
            logger.error("rss_exception", url=url, error=str(e))
            return None

    def _parse_entry_date(self, entry: Any) -> Optional[datetime]:
        for attr in ("published_parsed", "updated_parsed", "created_parsed"):
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    import calendar
                    ts = calendar.timegm(parsed)
                    return datetime.utcfromtimestamp(ts)
                except Exception:
                    continue
        return None

    def extract_entries(
        self,
        feed: feedparser.FeedParserDict,
        source: NewsSourceConfig,
        max_entries: int = 50,
    ) -> List[RawArticle]:
        articles: List[RawArticle] = []

        for entry in feed.entries[:max_entries]:
            url = getattr(entry, "link", None) or getattr(entry, "id", None)
            if not url:
                continue

            title = getattr(entry, "title", "") or ""
            # Content: prefer summary over description
            content = (
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
                or ""
            )

            # Strip HTML from content
            if content and "<" in content:
                content = BeautifulSoup(content, "lxml").get_text(separator=" ")

            title = re.sub(r"\s+", " ", title).strip()
            content = re.sub(r"\s+", " ", content).strip()

            published_at = self._parse_entry_date(entry)
            scraped_at = datetime.utcnow()

            article = RawArticle(
                url=url,
                title=title,
                content=content,
                source_name=source.name,
                published_at=published_at,
                scraped_at=scraped_at,
                reliability_score=source.reliability_score,
            )

            if article.is_arabic():
                articles.append(article)

        return articles



class ArticleScraper:


    CONTENT_SELECTORS = [
        "article.article-body",
        "div.article-content",
        "div.article-text",
        "div.wysiwyg",
        "div[class*='article']",
        "div[class*='content']",
        "div[class*='body']",
        "main",
        "article",
    ]

    def __init__(self, timeout: int = 15) -> None:
        self._cs = cloudscraper.create_scraper()
        self._timeout = timeout

    async def scrape(self, url: str) -> Optional[str]:
        try:
            with self._cs.get(
                url,
                timeout=self._timeout,
                allow_redirects=True,
            ) as response:
                if response.status_code != 200:
                    logger.warning("scrape_failed", url=url, status_code=response.status_code)
                    return None
                html = response.content
                return self._extract_text(html)
        except Exception as e:
            logger.warning("scrape_failed", url=url, error=str(e))
            return None

    def _extract_text(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup.find_all(
            ["script", "style", "nav", "header", "footer",
             "aside", "form", "button", "iframe", "figure", "noscript"]
        ):
            tag.decompose()
        for noise_term in [
            "sidebar", "menu", "footer", "header", "nav", "widget", "related",
            "social", "trending", "comments", "tags", "tag", "topic", "theme", 
            "category", "breadcrumb", "banner", "popular", "toolbar", "sharing",
            "b-menu", "b-header", "b-footer", "b-sidebar", "b-nav", "ad-", "popup"
        ]:
            for tag in soup.find_all(class_=re.compile(noise_term, re.IGNORECASE)):
                tag.decompose()
            for tag in soup.find_all(id=re.compile(noise_term, re.IGNORECASE)):
                tag.decompose()

        content_selectors = [
            ".article__text", ".b-article__text", "div.article-text",
            ".wysiwyg--all-content", "div.wysiwyg",
            "article.article-body", "div.article-content", "div.article-body",
            "article", "main"
        ]

        main_container = None
        for selector in content_selectors:
            container = soup.select_one(selector)
            if container:
                if len(container.get_text(strip=True)) > 200:
                    main_container = container
                    break

        if not main_container:
            main_container = soup.find("body") or soup

        paragraphs = main_container.find_all("p")
        if paragraphs:
            text_blocks = []
            for p in paragraphs:
                p_text = p.get_text(separator=" ", strip=True)
                if len(p_text) > 35:
                    text_blocks.append(p_text)
            
            combined = " ".join(text_blocks)
            if len(combined) > 150:
                return re.sub(r"\s+", " ", combined).strip()

        direct_text = main_container.get_text(separator=" ", strip=True)
        if len(direct_text) > 150:
            return re.sub(r"\s+", " ", direct_text).strip()

        return None