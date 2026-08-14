from __future__ import annotations
import asyncio
import logging
import sys
from collections import defaultdict
from sqlalchemy import select
from engine.synthesizer import NewsSynthesizer
from db.session import init_db, AsyncSessionLocal
from engine.manager import SourceManager
from models.orm import ArticleORM, ClusterORM
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineRunner")


async def run_intelligence_pipeline(
    search_query: str, 
    time_window: str = "3d", 
    limit: int = 10, 
    manager: SourceManager | None = None,
    synthesizer: NewsSynthesizer | None = None
):
    logger.info("Initializing SQLite database schemas...")
    await init_db()

    logger.info(f"Starting Google News search for query: '{search_query}' (Window: {time_window})")

    if manager is None:
        manager = SourceManager()
        
    if synthesizer is None:
        synthesizer = NewsSynthesizer()

    async with AsyncSessionLocal() as session:
        try:
            logger.info("Fetching and processing articles through the pipeline...")
            persisted_articles = await manager.search_google_news(
                query=search_query,
                time_window=time_window,
                limit=limit,
                scrape_full_content=True,
                session=session
            )

            if not persisted_articles:
                logger.warning("No new articles met the criteria or were retrieved.")
                return 

            logger.info(f"Processed and cached {len(persisted_articles)} articles successfully.")
            response = {"query": search_query, "time_window": time_window, "clusters": []}

            story_clusters = defaultdict(list)
            for article in persisted_articles:
                if getattr(article, "cluster_id", None):
                    story_clusters[article.cluster_id].append(article)

            logger.info(f"Identified {len(story_clusters)} active story clusters for this search.")

            for cluster_id, articles in story_clusters.items():
                cluster_data = {"cluster_id": cluster_id, "summary": None, "articles": []}
                print("\n" + "=" * 60)
                print(f"STORY CLUSTER #{cluster_id} ({len(articles)} Source Articles)")
                print("=" * 60)

                for idx, art in enumerate(articles, 1):
                    bias_percentage = round((1.0 - (art.neutrality_score or 0.0)) * 100, 1)
                    print(f"  {idx}. [{art.source_name}] {art.title}")
                    print(f"     └─ Heuristic Statement Type: {art.statement_type}")
                    print(f"     └─ Heuristic Attribution:    {art.attribution_label}")
                    print(f"     └─ Qwen Propaganda:          {art.propaganda_label}") 
                    print(f"     └─ Estimated Bias Percentage: {bias_percentage}%")
                    print(f"     └─ Reliability Score:        {art.reliability_score}")
                    print("-" * 50)

                articles_content = [art.content_clean or art.content or art.title for art in articles]
                summary = await synthesizer.synthesize_cluster(articles_content)
                cluster_data["summary"] = summary
                print("\n Neutral Summary ")
                print(summary)
                print("=" * 60 + "\n")
                
                response["clusters"].append(cluster_data)
                
            return response
            
        except Exception as e:
            logger.error(f"Pipeline execution encountered an exception: {e}", exc_info=True)
            await session.rollback()
        finally:
            await session.close()

if __name__ == "__main__":
    query_term = "اتفاق مكة"
    asyncio.run(run_intelligence_pipeline(search_query=query_term, time_window="5d", limit=10))
