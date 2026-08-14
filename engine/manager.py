from __future__ import annotations
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
import numpy as np
import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import NewsSourceConfig, settings
from core.sources_list import ARABIC_NEWS_SOURCES
from core.constants import ARABIC_STOPWORDS, ARABIC_BOILERPLATE_KEYWORDS, GEO_DOMAINS, IGNORE_TITLE_WORDS
from engine.fetcher import ArticleScraper, RSSFetcher, GoogleNews
from engine.scraper import RawArticle
from models.orm import ArticleORM, ClusterORM
from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from preprocessing.tokenizer import ArabicTokenizer, ArabicStopwordFilter
from preprocessing.deduplicator import ArticleDeduplicator
from preprocessing.analyzer import HeuristicScorer, StoryGrouper, AraBERTClassifier
from preprocessing.utils import is_arabic_text
from preprocessing.ner import NER

logger = structlog.get_logger(__name__)


def _light_stem(word: str) -> str:
    word = word.replace("ة", "ه").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    for prefix in ["ال", "وال", "بال", "فال", "لل"]:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            word = word[len(prefix):]
            break
    for suffix in ["ين", "ون", "ات", "يه", "ية", "ها", "هم", "نا", "ي"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            word = word[:-len(suffix)]
            break
    return word


def _share_entities(set_a: Set[str], set_b: Set[str]) -> bool:
    if not set_a or not set_b:
        return False
    if set_a & set_b:
        return True
    
    for item_a in set_a:
        clean_a = item_a.replace("ال", "").strip()
        for item_b in set_b:
            clean_b = item_b.replace("ال", "").strip()
            if len(clean_a) > 3 and len(clean_b) > 3:
                if clean_a in clean_b or clean_b in clean_a:
                    return True
    return False


def is_boilerplate(text: str) -> bool:
    if not text:
        return True
    if len(text) < 250:
        match_count = sum(1 for kw in ARABIC_BOILERPLATE_KEYWORDS if kw in text)
        if match_count >= 3:
            return True
    return False


def has_geographical_conflict(locs_a: List[str], locs_b: List[str], title_a: str, title_b: str) -> bool:
    set_a = {l.lower() for l in locs_a} | set(title_a.lower().split())
    set_b = {l.lower() for l in locs_b} | set(title_b.lower().split())

    for domain_name, terms in GEO_DOMAINS.items():
        has_a = any(term in set_a for term in terms)
        if not has_a:
            continue
            
        for other_domain, other_terms in GEO_DOMAINS.items():
            if other_domain == domain_name:
                continue
            has_b = any(term in set_b for term in other_terms)
            if has_b:
                has_a_other = any(term in set_a for term in other_terms)
                has_b_this = any(term in set_b for term in terms)
                if not has_a_other and not has_b_this:
                    return True
                    
    return False


def has_title_vocabulary_conflict(title_a: str, title_b: str) -> bool:
    words_a = {w.strip('"«»“”\'').strip(':') for w in title_a.lower().split() if len(w) > 2}
    words_b = {w.strip('"«»“”\'').strip(':') for w in title_b.lower().split() if len(w) > 2}
    
    action_words_a = words_a - IGNORE_TITLE_WORDS
    action_words_b = words_b - IGNORE_TITLE_WORDS
    
    if not action_words_a or not action_words_b:
        return False
        
    intersection = action_words_a & action_words_b
    if not intersection:
        stems_a = {_light_stem(w) for w in action_words_a}
        stems_b = {_light_stem(w) for w in action_words_b}
        if not (stems_a & stems_b):
            return True 
            
    return False


def _extract_newspaper_content(url: str) -> str:
    try:
        from newspaper import Article
        article = Article(url, language='ar')
        article.download()
        article.parse()
        return article.text or ""
    except Exception:
        return ""


class SourceManager:
    def __init__(self) -> None:
        self._sources = ARABIC_NEWS_SOURCES
        self._cleaner = ArabicNewsCleaner(remove_numbers=False, keep_quotes=True)
        self._normalizer = ArabicNormalizer()
        self._tokenizer = ArabicTokenizer()
        self._stop_filter = ArabicStopwordFilter(extra_stopwords=ARABIC_STOPWORDS)
        
        self._scorer = HeuristicScorer()
        self._classifier = AraBERTClassifier()
        self._grouper = StoryGrouper()
        
        self._ner = None
        self._deduplicator = ArticleDeduplicator(similarity_threshold=0.70)
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
                self._deduplicator.seed_known_article(art.url, art.content)
            elif art.title:
                self._deduplicator.seed_known_article(art.url, art.title)

            if art.cluster_id and art.content_clean:
                self._grouper.seed_cluster_anchor(art.cluster_id, art.content_clean)

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
                logger.error("source_fetch_failed", source=self._sources[i].name, error=str(result))
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
        query: Optional[str] = None,

    ) -> List[ArticleORM]:
        if not raw_articles:
            return []

        unique_raw_map = {a.url: a for a in raw_articles}
        urls = list(unique_raw_map.keys())
        
        existing_stmt = select(ArticleORM.url).where(ArticleORM.url.in_(urls))
        existing_result = await session.execute(existing_stmt)
        existing_urls = {row[0] for row in existing_result.fetchall()}

        time_cutoff = datetime.utcnow() - timedelta(hours=48)
        active_stmt = select(ArticleORM).where(
            and_(
                ArticleORM.published_at >= time_cutoff,
                ArticleORM.cluster_id.is_not(None)
            )
        )
        active_result = await session.execute(active_stmt)
        active_articles = list(active_result.scalars().all())



        # max_id_stmt = select(ArticleORM.cluster_id).order_by(ArticleORM.cluster_id.desc()).limit(1)
        # max_id_res = await session.execute(max_id_stmt)
        # max_id_row = max_id_res.fetchone()
        # next_cluster_id = (max_id_row[0] + 1) if max_id_row and max_id_row[0] else 1


        #############################################################################################3

        existing_clusters_stmt = select(ClusterORM)
        existing_clusters_result = await session.execute(existing_clusters_stmt)
        existing_clusters = {cluster.id: cluster for cluster in existing_clusters_result.scalars().all()}
        #########################################################################################

        new_articles: List[ArticleORM] = []

        embedding_cache: Dict[int, np.ndarray] = {}
        for art in active_articles:
            art_text = art.title_clean + " " + (art.content_clean[:400] if art.content_clean else "")
            embedding_cache[art.id] = self._grouper.get_normalized_embedding(art_text)


        valid_items = []
        for url, raw in unique_raw_map.items():
            if url in existing_urls:
                continue
            
            if not is_arabic_text(raw.title + " " + (raw.content or "")):
                continue

            content_to_use = raw.content or ""
            if is_boilerplate(content_to_use):
                content_to_use = raw.title

            cleaned_title = self._cleaner.clean(raw.title)
            normalized_title = self._normalizer.normalize(cleaned_title)

            cleaned_content = self._cleaner.clean(content_to_use)
            normalized_content = self._normalizer.normalize(cleaned_content)
            
            is_dup, _ = self._deduplicator.is_duplicate(raw.url, normalized_content)
            if is_dup:
                continue

            existing_urls.add(url)

            tokens = self._tokenizer.tokenize(normalized_content)
            filtered_tokens = self._stop_filter.filter(tokens)

            from preprocessing.propaganda_features import calculate_loaded_words_ratio
            loaded_words_ratio = calculate_loaded_words_ratio(filtered_tokens)

            valid_items.append({
                "raw": raw,
                "content_to_use": content_to_use,
                "normalized_title": normalized_title,
                "normalized_content": normalized_content,
                "cleaned_content": cleaned_content,
                "filtered_tokens": filtered_tokens,
                "loaded_words_ratio": loaded_words_ratio,
            })

        if not valid_items:
            return []
        ner_texts = [item["cleaned_content"][:800] for item in valid_items]
        batch_entities = [{"person": [], "location": [], "organization": [], "misc": []} for _ in valid_items]


        qwen_input_items = [
            {
                "title": item["raw"].title,
                "text": item["normalized_content"][:800],
                "loaded_words_ratio": item["loaded_words_ratio"]
            }
            for item in valid_items
        ]
        batch_propaganda = self._classifier.classify_propaganda_batch(qwen_input_items)

        for idx, item in enumerate(valid_items):
            raw = item["raw"]
            content_to_use = item["content_to_use"]
            normalized_title = item["normalized_title"]
            normalized_content = item["normalized_content"]
            filtered_tokens = item["filtered_tokens"]

            entities = batch_entities[idx]
            predicted_propaganda = batch_propaganda[idx]

            new_persons = set(entities.get("person", []))
            new_locations = set(entities.get("location", []))
            new_orgs = set(entities.get("organization", []))
            
            text_to_embed = normalized_title + " " + normalized_content[:400]
            new_norm_emb = self._grouper.get_normalized_embedding(text_to_embed)
            
            candidates: List[ArticleORM] = []
            candidate_confidences: Dict[int, float] = {}
            
            new_title_stems = {_light_stem(w) for w in normalized_title.split() if len(w) > 3}

            for active_art in active_articles:
                art_locations = active_art.locations or []
                
                if has_geographical_conflict(
                    locs_a=list(new_locations),
                    locs_b=art_locations,
                    title_a=raw.title,
                    title_b=active_art.title
                ):
                    continue

                if has_title_vocabulary_conflict(raw.title, active_art.title):
                    continue

                art_persons = set(active_art.persons or [])
                art_orgs = set(active_art.organizations or [])

                has_person_match = _share_entities(new_persons, art_persons)
                has_location_match = _share_entities(new_locations, set(art_locations))
                has_org_match = _share_entities(new_orgs, art_orgs)

                art_title_stems = {_light_stem(w) for w in (active_art.title_clean or "").split() if len(w) > 3}
                stem_intersection = new_title_stems & art_title_stems

                matches_count = sum([has_person_match, has_location_match, has_org_match])
                title_weight = min(1.0, len(stem_intersection) / 4.0)
                confidence = (matches_count * 0.35) + (title_weight * 0.5)
                
                candidates.append(active_art)
                candidate_confidences[active_art.id] = confidence

            assigned_cluster_id = None
            base_threshold = 0.87 

            if candidates:
                cluster_groups = defaultdict(list)
                for cand in candidates:
                    cluster_groups[cand.cluster_id].append(cand)

                best_cluster_id = None
                best_similarity = -1.0
                best_group_confidence = 0.0

                for cid, group in cluster_groups.items():
                    group_embeddings = []
                    max_heuristic_confidence = 0.0

                    for art in group:
                        art_emb = embedding_cache.get(art.id)
                        if art_emb is None:
                            art_text = art.title_clean + " " + (art.content_clean[:400] if art.content_clean else "")
                            art_emb = self._grouper.get_normalized_embedding(art_text)
                            embedding_cache[art.id] = art_emb
                            
                        group_embeddings.append(art_emb)
                        
                        conf = candidate_confidences.get(art.id, 0.0)
                        if conf > max_heuristic_confidence:
                            max_heuristic_confidence = conf

                    cluster_centroid = np.mean(group_embeddings, axis=0)
                    centroid_norm = np.linalg.norm(cluster_centroid)
                    if centroid_norm > 0:
                        cluster_centroid /= centroid_norm

                    similarity = float(np.dot(new_norm_emb, cluster_centroid))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster_id = cid
                        best_group_confidence = max_heuristic_confidence

                if best_group_confidence >= 0.7:
                    adapted_threshold = 0.83  
                elif best_group_confidence >= 0.4:
                    adapted_threshold = 0.85  
                else:
                    adapted_threshold = base_threshold

                if best_similarity >= adapted_threshold:
                    assigned_cluster_id = best_cluster_id

            if assigned_cluster_id is None:
                # assigned_cluster_id = next_cluster_id
                # next_cluster_id += 1
                #######################################################3
                new_cluster = ClusterORM(query=query)
                session.add(new_cluster)
                await session.flush()
                assigned_cluster_id = new_cluster.id
                existing_clusters[assigned_cluster_id] = new_cluster
###########################################################################33

            scores = self._scorer.evaluate_article(
                source_name=raw.source_name,
                raw_text=content_to_use[:1000],
                tokens=filtered_tokens,
                title=raw.title
            )

            computed_attribution = self._scorer.determine_attribution_label(scores["attribution_score"])
            computed_statement = self._scorer.determine_statement_type(
                title=raw.title, 
                raw_text=content_to_use[:1000], 
                neutrality_score=scores["neutrality_score"]
            )

            is_verified = (
                scores["reliability_score"] >= 0.70 and 
                scores["neutrality_score"] >= 0.60 and          
                computed_statement == "reporting" and           
                computed_attribution == "supported_claim" and 
                predicted_propaganda.lower() in ("neutral", "no_propaganda")
            )
            article_orm = ArticleORM(
                url=raw.url,
                source_name=raw.source_name,
                title=raw.title,
                content=content_to_use,
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
                
                propaganda_label=predicted_propaganda,
                statement_type=computed_statement,
                attribution_label=computed_attribution,
                verified=is_verified,

                published_at=raw.published_at,
                scraped_at=raw.scraped_at,
                language=raw.language,
                word_count=len(content_to_use.split()),
                is_processed=True,
            )
            session.add(article_orm)
            new_articles.append(article_orm)
            active_articles.append(article_orm)
            
            embedding_cache[article_orm.id] = new_norm_emb

        try:
            await session.flush()
            logger.info("articles_persisted", new_count=len(new_articles))
        except Exception as e:
            logger.error("persistence_error", error=str(e))
            await session.rollback()
            return []

        return new_articles
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
        limit: int = 10, 
        scrape_full_content: bool = True,
        session: Optional[AsyncSession] = None
    ) -> List[Any]:
        
        google_news = GoogleNews(query=query, time_window=time_window, limit=limit, scrape_full_content=scrape_full_content)
        entries = await google_news.fetch_news()
        
        scraper = ArticleScraper(timeout=3)
        articles: List[RawArticle] = []
        semaphore = asyncio.Semaphore(15)

        async def fetch_single_article(entry):
            async with semaphore:
                try:
                    title = getattr(entry, "title", "") or ""
                    if " - " in title:
                        title = title.rsplit(" - ", 1)[0]

                    text = None
                    try:
                        text = await asyncio.wait_for(scraper.scrape(entry.link), timeout=3.5)
                    except Exception:
                        pass  
                    
                    if not text or len(text) < 50:
                        raw_summary = getattr(entry, "summary", getattr(entry, "description", title))
                        from bs4 import BeautifulSoup
                        text = BeautifulSoup(raw_summary, "lxml").get_text(separator=" ").strip()

                    source_name = getattr(entry.source, "title", "Google News") if hasattr(entry, "source") else "Google News"
                    
                    pub_date = datetime.utcnow()
                    if hasattr(entry, "published"):
                        try:
                            from dateutil import parser as dt_parser
                            pub_date = dt_parser.parse(entry.published)
                            if pub_date.tzinfo is not None:
                                pub_date = pub_date.astimezone(timezone.utc).replace(tzinfo=None)
                        except Exception:
                            pass

                    return RawArticle(
                        url=entry.link,
                        title=title,
                        content=text or title,
                        source_name=source_name,
                        published_at=pub_date,
                        scraped_at=datetime.utcnow(),
                        reliability_score=0.70,
                        language="ar",
                    )
                except Exception as e:
                    logger.error("google_news_article_error", url=getattr(entry, "link", ""), error=str(e))
                    return None

        tasks = [fetch_single_article(e) for e in entries[:limit]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if res and isinstance(res, RawArticle):
                articles.append(res)

        if session and articles:
            persisted_articles = await self._persist_articles(session=session, raw_articles=articles, query=query)
            return persisted_articles 
            
        return articles