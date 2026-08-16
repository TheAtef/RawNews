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
import gc
import torch
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
    # synthesizer: NewsSynthesizer | None = None
):

    logger.info(f"Starting Google News search for query: '{search_query}' (Window: {time_window})")

    if manager is None:
        manager = SourceManager()
        
    # if synthesizer is None:
    #     synthesizer = NewsSynthesizer()

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
                            cluster_data["articles"] = articles  

                            articles_content = [art.content_clean or art.content or art.title for art in articles]
                            # summary = await synthesizer.synthesize_cluster(articles_content)
                            # cluster_data["summary"] = summary
                            # cluster = await session.get(ClusterORM, cluster_id)

                            # if cluster:
                            #     cluster.summary = summary

                            # print("\n Neutral Summary ")
                            # print(summary)
                            cluster_data["summary"]=None
                            print("\n" + "=" * 60)
                            print("=" * 60 + "\n")
                            
                            response["clusters"].append(cluster_data)
                            await session.commit()
            return response
            
        except Exception as e:
            logger.error(f"Pipeline execution encountered an exception: {e}", exc_info=True)
            await session.rollback()
        finally:
            await session.close()



async def generate_cluster_summary(cluster_id: int,synthesizer: NewsSynthesizer | None = None):
    logger.info(f"Starting summary generation for cluster {cluster_id}")
    await init_db()
    
    logger.info("Database initialized")
    if synthesizer is None:
        synthesizer = NewsSynthesizer()
    async with AsyncSessionLocal() as session:
        try:
            cluster = await session.get(ClusterORM, cluster_id)

            if cluster is None:
                return None

            result = await session.execute(select(ArticleORM).where(ArticleORM.cluster_id == cluster_id))
            articles = result.scalars().all()

            if not articles:
                return None
            articles_content = [article.content_clean or article.content or article.title for article in articles]


            if not articles_content:
                return None
            logger.info(f"Starting Gemma summarization "f"for cluster {cluster_id}...")
            logger.info(
                f"Gemma state | "
                f"use_local={synthesizer.use_local} | "
                f"is_enabled={synthesizer.is_enabled} | "
                f"model_loaded={synthesizer.model is not None} | "
                f"tokenizer_loaded={synthesizer.tokenizer is not None}"
            )
            summary = await synthesizer.synthesize_cluster(articles_content)
            logger.info(f"Gemma summarization completed "f"for cluster {cluster_id}")

            logger.info(f"Summary : {summary} ")
            cluster.summary = summary
            await session.commit()

            return {
                "cluster_id": cluster_id,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Cluster summary generation failed: {e}",exc_info=True)
            await session.rollback()
            return None

        finally:
            await session.close()
            if synthesizer is not None:
                synthesizer.model = None
                synthesizer.tokenizer = None
                synthesizer.is_enabled = False

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Gemma resources released for cluster {cluster_id}")
if __name__ == "__main__":
    # query_term = "اتفاق مكة"
    # asyncio.run(run_intelligence_pipeline(search_query=query_term, time_window="5d", limit=10))
    asyncio.run(generate_cluster_summary(cluster_id=1))

