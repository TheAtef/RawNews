# test_live_search_pipeline.py
import asyncio
import sys
from sqlalchemy import select

from db.session import AsyncSessionLocal, init_db, drop_db
from engine.manager import SourceManager
from models.orm import ArticleORM


async def run_live_search_pipeline(query: str):
    print("1. Re-initializing database schemas to ensure a clean slate...")
    await drop_db()
    await init_db()
    
    manager = SourceManager()
    
    print("\n2. Launching live scraper across internet sources...")
    print("Connecting to RSS feeds, parsing HTML content, and running preprocessing pipeline...")
    print("This may take 15-30 seconds depending on network connections...")
    
    async with AsyncSessionLocal() as session:
        try:
            scraped_articles = await manager.fetch_all(
                session=session,
                max_per_source=10,
                age_hours=24, 
                scrape_full_content=True
            )
            await session.commit()
            print(f"\nScraping complete. Preprocessed and saved {len(scraped_articles)} live articles.")
        except Exception as e:
            print(f"Failed to scrape data from internet: {e}")
            await session.rollback()
            return

        if not scraped_articles:
            print("No articles were successfully crawled. Please check your internet connection or feed status.")
            return

        print(f"\n3. Preprocessing search query and searching matching text for: '{query}'...")
        matched_articles = await manager.search_articles(
            session=session,
            query=query,
            time_window_hours=24,
            limit=10
        )

        if not matched_articles:
            print(f"\nNo articles matching the query '{query}' were found in the live crawled batch.")
            
            print("\nSuggested keywords from this crawling batch:")
            sample_stmt = select(ArticleORM.title_clean).limit(5)
            sample_titles = (await session.execute(sample_stmt)).scalars().all()
            for title in sample_titles:
                words = title.split()
                if len(words) > 2:
                    print(f" - {' '.join(words[:3])}")
            return

        seed_article = matched_articles[0]
        cluster_id = seed_article.cluster_id

        print("\n" + "=" * 80)
        print("SEED ARTICLE DETAILS (SCRAPED REAL-TIME):")
        print(f"ID:           {seed_article.id}")
        print(f"Source:       {seed_article.source_name}")
        print(f"Original:     {seed_article.title}")
        print(f"Normalized:   {seed_article.title_clean}")
        print(f"Cluster ID:   {seed_article.cluster_id}")
        print(f"Reliability:  {seed_article.reliability_score}")
        print("=" * 80)

        if cluster_id is None:
            print("\nThis matching article does not belong to any cluster (no similar reports were found).")
            return

        print(f"\n4. Retrieving similar articles grouped under Cluster ID #{cluster_id}...")
        cluster_stmt = (
            select(ArticleORM)
            .where(ArticleORM.cluster_id == cluster_id)
            .order_by(ArticleORM.reliability_score.desc())
        )
        cluster_result = await session.execute(cluster_stmt)
        clustered_articles = cluster_result.scalars().all()

        print(f"Found {len(clustered_articles)} matched reports in this story group:")

        for i, art in enumerate(clustered_articles):
            is_seed = " [Target Match]" if art.id == seed_article.id else ""
            role = "Anchor Report" if i == 0 else "Related Report"
            
            print("\n" + f"  [{role}]{is_seed}")
            print(f"  ID:          {art.id}")
            print(f"  Source:      {art.source_name}")
            print(f"  Title:       {art.title}")
            print(f"  Reliability: {art.reliability_score} | Neutrality: {art.neutrality_score} | Attribution: {art.attribution_score}")
            print(f"  Preview:     {art.content_clean[:180] if art.content_clean else 'No clean content available'}...")
            print("  " + "-" * 70)


if __name__ == "__main__":
    search_keyword = "الشرع"
    
    if len(sys.argv) > 1:
        search_keyword = " ".join(sys.argv[1:])

    asyncio.run(run_live_search_pipeline(search_keyword))