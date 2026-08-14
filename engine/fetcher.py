import asyncio
from datetime import datetime
import re
from typing import Any, List, Optional
import structlog
from bs4 import BeautifulSoup
import feedparser
import cloudscraper
import json
import requests
import httpx
from core.config import NewsSourceConfig, settings
from engine.scraper import RawArticle

logger = structlog.get_logger(__name__)


class RSSFetcher:
    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout
        self._cs = cloudscraper.create_scraper()

    async def fetch_feed(self, url: str) -> Optional[feedparser.FeedParserDict]:
        try:
            response = await asyncio.to_thread(self._cs.get, url, timeout=self._timeout)
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
            content = (
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
                or ""
            )

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

class GoogleNews:
    def __init__(self, query: str, time_window: str = "7d", limit: int = 20, scrape_full_content: bool = True):
        self.cs = cloudscraper.create_scraper()
        self.query = query
        self.time_window = time_window
        self.limit = limit
        self.scrape_full_content = scrape_full_content
        self.link = f'https://news.google.com/rss/search?q={query}when%3A{time_window}&hl=ar&gl=SA&ceid=SA%3Aar'
    
    async def fetch_news(self) -> List[feedparser.FeedParserDict]:
        response = await asyncio.to_thread(self.cs.get, self.link, timeout=15)
        articles = []
        if response.status_code == 200:
            rss = feedparser.parse(response.content)
            logger.info("google_news_fetch_success", query=self.query, total_entries=len(rss.entries))
            
            entries_to_process = rss.entries[:self.limit]
            semaphore = asyncio.Semaphore(20)  

            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as http_client:
                async def resolve_entry(entry):
                    async with semaphore:
                        try:
                            url = await self.getGoogleRedirectUrl(http_client, entry.link)
                            if any(blocked in url for blocked in settings.blocked_websites):
                                return None
                            entry.link = url
                            return entry
                        except Exception as e:
                            logger.warning("google_redirect_failed", link=getattr(entry, "link", ""), error=str(e))
                            return entry 

                tasks = [resolve_entry(e) for e in entries_to_process]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for res in results:
                    if res and not isinstance(res, Exception):
                        articles.append(res)
                    if len(articles) >= self.limit:
                        break
        return articles

    async def getGoogleRedirectUrl(self, client: httpx.AsyncClient, rss_url: str) -> str:
        try:
            resp = await client.get(rss_url)
            if resp.status_code != 200:
                return rss_url
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            cwiz = soup.select_one('c-wiz[data-p]')
            if not cwiz or not cwiz.get('data-p'):
                return str(resp.url) or rss_url  
                
            data = cwiz.get('data-p')
            obj = json.loads(data.replace('%.@.', '["garturlreq",'))

            payload = {
                'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])
            }

            headers = {
                'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }

            url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
            response = await client.post(url, headers=headers, data=payload)
            
            if response.status_code == 200:
                array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
                article_url = json.loads(array_string)[1]
                return article_url
            return rss_url
        except Exception:
            return rss_url
    


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
            response = await asyncio.to_thread(
                self._cs.get, url, timeout=self._timeout, allow_redirects=True
            )
            if response.status_code != 200:
                logger.warning("scrape_failed", url=url, status_code=response.status_code)
                return None
            html = response.content
            return self._extract_text(html)
        except Exception as e:
            logger.warning("scrape_failed", url=url, error=str(e))
            return None

    def _extract_text(self, html: bytes) -> Optional[str]:
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