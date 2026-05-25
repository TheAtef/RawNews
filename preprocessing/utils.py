from __future__ import annotations
import re

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", re.UNICODE)
RE_MENTION = re.compile(r"@\w+", re.UNICODE)
RE_HASHTAG = re.compile(r"#\w+", re.UNICODE)
RE_NUMBER = re.compile(r"\d+", re.UNICODE)

RE_PUNCT_PRESERVE_QUOTES = re.compile(
    r"[^\w\s\"'«»“”]", 
    re.UNICODE
)

RE_REPEAT = re.compile(r"(.)\1{2,}", re.UNICODE)
RE_WS = re.compile(r"\s+", re.UNICODE)

ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EXTENDED_INDIC_MAP = str.maketrans("۰۱۲۳۴۵۶۷٨٩", "0123456789")

def is_arabic_text(text: str, threshold: float = 0.40) -> bool:
    if not text or not isinstance(text, str):
        return False

    arabic_char_count = sum(
        1 for character in text
        if "\u0600" <= character <= "\u06FF"
        or "\u0750" <= character <= "\u077F"
        or "\uFB50" <= character <= "\uFDFF"
        or "\uFE70" <= character <= "\uFEFF"
    )
    total_characters = len(text)
    if total_characters == 0:
        return False
        
    return (arabic_char_count / total_characters) >= threshold