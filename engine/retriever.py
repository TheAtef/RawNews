from models.orm import ArticleORM
from typing import List
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta

from preprocessing.cleaner import ArabicNewsCleaner
from preprocessing.normalizer import ArabicNormalizer
from preprocessing.tokenizer import ArabicTokenizer, ArabicStopwordFilter
from core.constants import ARABIC_STOPWORDS

class Retriever:
    def __init__(self) -> None:
        self._cleaner = ArabicNewsCleaner(remove_numbers=False, keep_quotes=True)
        self._normalizer = ArabicNormalizer()
        self._tokenizer = ArabicTokenizer()
        self._stop_filter = ArabicStopwordFilter(extra_stopwords=ARABIC_STOPWORDS)
            
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