import asyncio
import structlog

from db.session import AsyncSessionLocal
from engine.manager import SourceManager
from core.config import settings

logger = structlog.get_logger(__name__)

_background_fetch_running = False
source_manager = SourceManager()



async def background_fetch_loop(interval_minutes: int = 30):
    global _background_fetch_running

    _background_fetch_running = True

    while _background_fetch_running:
        try:
            logger.info("background_fetch_starting")

            async with AsyncSessionLocal() as session:

                articles = await source_manager.fetch_all(
                    session=session,
                    max_per_source=settings.max_articles_per_source,
                    age_hours=settings.article_max_age_hours,
                    scrape_full_content=True,
                )

                await session.commit()

                logger.info(
                    "background_fetch_done",
                    new_articles=len(articles)
                )

        except Exception as e:
            logger.error(
                "background_fetch_error",
                error=str(e)
            )

        await asyncio.sleep(interval_minutes * 60)


def stop_background_fetch():
    global _background_fetch_running
    _background_fetch_running = False