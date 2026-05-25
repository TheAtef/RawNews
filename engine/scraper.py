from datetime import datetime

from typing import Optional


class RawArticle:

    __slots__ = (
        "url", "title", "content", "source_name", "published_at", "scraped_at",
        "reliability_score", "language",
    )

    def __init__(
        self,
        url: str,
        title: str,
        content: str,
        source_name: str,
        published_at: Optional[datetime],
        scraped_at: Optional[datetime],
        reliability_score: float,
        language: str = "ar",
    ) -> None:
        self.url = url
        self.title = title
        self.content = content
        self.source_name = source_name
        self.published_at = published_at
        self.scraped_at = scraped_at
        self.reliability_score = reliability_score
        self.language = language

    def is_arabic(self, min_ratio: float = 0.40) -> bool:
        text = self.title + " " + (self.content or "")
        if not text.strip():
            return False
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return arabic_chars / len(text) >= min_ratio

    def __repr__(self) -> str:
        return f"<RawArticle source={self.source_name!r} title={self.title[:40]!r}>"

