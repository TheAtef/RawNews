# search_pipeline.py
from __future__ import annotations

import collections
import collections.abc

from engine.retriever import Retriever

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

import asyncio
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.config import settings
from db.base import Base
from engine.manager import SourceManager
from engine.synthesizer import NewsSynthesizer

logger = structlog.get_logger()


async def search_and_synthesize(keyword: str, time_window:str, limit:int) -> None:
    time_hours = int(time_window[:-1]) * 24 if "d" in time_window else int(time_window[:-1])
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    retriever = Retriever()
    synthesizer = NewsSynthesizer()

    async with async_session() as session:
        logger.info("searching_local_database", keyword=keyword)
        local_articles = await retriever.search_articles(session, query=keyword, time_window_hours=time_hours, limit=limit)

        if len(local_articles) < 3:
            manager = SourceManager()
            logger.info("insufficient_local_articles_fetching_live", keyword=keyword)
            await manager.search_google_news(
                query=keyword,
                time_window=time_window,
                limit=limit,
                scrape_full_content=True,
                session=session
            )
            await session.commit()

            local_articles = await retriever.search_articles(session, query=keyword, time_window_hours=time_hours, limit=limit)

    if not local_articles:
        logger.warn("no_articles_found_for_keyword", keyword=keyword)
        await engine.dispose()
        return

    logger.info("analyzing_search_results", count=len(local_articles))
    from collections import defaultdict
    clusters = defaultdict(list)
    [clusters[art.cluster_id].append(art) for art in local_articles]
    print(f"\n================📊 ANALYSIS RESULTS FOR: '{keyword}' ================")
    for cluseter, articles in clusters.items():
        print(f"\n================📊 Cluseter: '{cluseter}' ================")
        articles_content = []
        for idx, art in enumerate(articles):
            text_to_summarize = art.content if art.content else art.title
            articles_content.append(text_to_summarize)

            print(f"\n[{idx + 1}] العنوان: {art.title}")
            print(f"    المصدر: {art.source_name} | الموثوقية: {art.reliability_score:.2f}")
            print(f"    الرابط: {art.url}")
            print(f"    تحليل التضليل (Propaganda): {art.propaganda_label}")
            print(f"    نوع الصياغة (Statement Type): {art.statement_type}")
            print(f"    حالة الإسناد (Attribution): {art.attribution_label}")
            print(f"    الكيانات المستخرجة: أشخاص={art.persons[:3] if art.persons else []}, أماكن={art.locations[:3] if art.locations else []}")
            print(f"    Cluster: {art.cluster_id}")
        print("======================================================================\n")

        logger.info("generating_neutral_synthesis", source_count=len(articles_content))
        synthesis = await synthesizer.synthesize_cluster(articles_content)

        if synthesis:
            print(f"================📝 NEUTRAL SUMMARY FOR CLUSTER {cluseter} ================")
            print(synthesis)
            print("=============================================================\n")
        else:
            logger.error("synthesis_generation_failed")

    await engine.dispose()


if __name__ == "__main__":
    import sys
    query_word = "سوريا"
    if len(sys.argv) > 1:
        query_word = " ".join(sys.argv[1:])
        
    asyncio.run(search_and_synthesize(keyword=query_word, time_window="7d", limit=15))