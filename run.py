# run_pipeline.py
from __future__ import annotations
import asyncio
import logging
import sys
from collections import defaultdict
from sqlalchemy import select
from engine.synthesizer import NewsSynthesizer

from db.session import init_db, AsyncSessionLocal, engine
from engine.manager import SourceManager
from models.orm import ArticleORM

# Setup standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineRunner")


async def run_intelligence_pipeline(search_query: str, time_window: str = "3d", limit: int = 10):
    logger.info("Initializing SQLite database schemas...")
    await init_db()

    logger.info(f"Starting Google News search for query: '{search_query}' (Window: {time_window})")
    
    manager = SourceManager()
    
    synthesizer = NewsSynthesizer()

    async with AsyncSessionLocal() as session:
        try:
            logger.info("Fetching and processing articles through the pipeline...")
            raw_articles = await manager.search_google_news(
                query=search_query,
                time_window=time_window,
                limit=limit,
                scrape_full_content=True,
                session=session
            )

            if not raw_articles:
                logger.warning("No new articles met the criteria or were retrieved.")
                return 

            logger.info(f"Processed and cached {len(raw_articles)} raw articles successfully.")

            stmt = select(ArticleORM).where(ArticleORM.cluster_id.is_not(None))
            result = await session.execute(stmt)
            persisted_articles = result.scalars().all()

            story_clusters = defaultdict(list)
            for article in persisted_articles:
                story_clusters[article.cluster_id].append(article)

            logger.info(f"Identified {len(story_clusters)} active story clusters in local database.")

            for cluster_id, articles in story_clusters.items():
                            print("\n" + "=" * 60)
                            print(f"STORY CLUSTER #{cluster_id} ({len(articles)} Source Articles)")
                            print("=" * 60)
                            
                            articles_content = []
                            
                            for idx, art in enumerate(articles, 1):
                                bias_percentage = round((1.0 - art.neutrality_score) * 100, 1)

                                print(f"  {idx}. [{art.source_name}] {art.title}")
                                print(f"     └─ Heuristic Statement Type: {art.statement_type}")
                                print(f"     └─ Heuristic Attribution:    {art.attribution_label}")
                                print(f"     └─ Qwen Propaganda:          {art.propaganda_label}") 
                                print(f"     └─ Estimated Bias Percentage: {bias_percentage}%")
                                print(f"     └─ Reliability Score:        {art.reliability_score}")
                                print("-" * 50)
                                
                                articles_content.append(art.content_clean or art.content or art.title)
                            
                            summary = await synthesizer.synthesize_cluster(articles_content)
                            print("\n Neutral Summary ")
                            print(summary)
                            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Pipeline execution encountered an exception: {e}", exc_info=True)
            await session.rollback()
        finally:
            await session.close()
            await engine.dispose()


if __name__ == "__main__":
    query_term = "اتفاق مكة"
    
    asyncio.run(run_intelligence_pipeline(search_query=query_term, time_window="5d", limit=8))