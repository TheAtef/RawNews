from __future__ import annotations
import re
from bs4 import BeautifulSoup
from preprocessing.utils import (
    RE_URL, RE_EMAIL, RE_MENTION, RE_HASHTAG, RE_NUMBER, 
    RE_PUNCT_PRESERVE_QUOTES, RE_REPEAT, RE_WS,
    ARABIC_INDIC_MAP, EXTENDED_INDIC_MAP
)

class ArabicNewsCleaner:
    def __init__(
        self,
        remove_numbers: bool = False,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        keep_quotes: bool = True,
    ) -> None:
        self.remove_numbers = remove_numbers
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.keep_quotes = keep_quotes

    def clean(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = self.remove_html(text)
        text = RE_URL.sub(" ", text)
        text = RE_EMAIL.sub(" ", text)

        if self.remove_mentions:
            text = RE_MENTION.sub(" ", text)
        if self.remove_hashtags:
            text = RE_HASHTAG.sub(" ", text)

        text = self.normalise_digits(text)
        if self.remove_numbers:
            text = RE_NUMBER.sub(" ", text)

        if self.keep_quotes:
            text = RE_PUNCT_PRESERVE_QUOTES.sub(" ", text)
        else:
            text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

        text = RE_REPEAT.sub(r"\1\1", text)
        text = RE_WS.sub(" ", text)

        return text.strip()

    @staticmethod
    def remove_html(text: str) -> str:
        try:
            return BeautifulSoup(text, "lxml").get_text(separator=" ")
        except Exception:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ")

    @staticmethod
    def normalise_digits(text: str) -> str:
        return text.translate(ARABIC_INDIC_MAP).translate(EXTENDED_INDIC_MAP)