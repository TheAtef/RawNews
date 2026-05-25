from __future__ import annotations
import re
import unicodedata

_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]",
    re.UNICODE
)
_TATWEEL = re.compile(r"\u0640", re.UNICODE)

_ALEF_MAP = str.maketrans(
    "\u0622\u0623\u0625\u0671\u0672\u0673\u0674\u0675",
    "\u0627" * 8
)
_YA_MAP = str.maketrans("\u0649", "\u064A")
_TA_MARBUTA_MAP = str.maketrans("\u0629", "\u0647")

class ArabicNormalizer:
    def __init__(
        self,
        normalize_alef: bool = True,
        normalize_ya: bool = False,        
        normalize_ta_marbuta: bool = False,
        remove_diacritics: bool = True,     
        remove_tatweel: bool = True,
        normalize_unicode: bool = True,
    ) -> None:
        self.normalize_alef = normalize_alef
        self.normalize_ya = normalize_ya
        self.normalize_ta_marbuta = normalize_ta_marbuta
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_unicode = normalize_unicode

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.remove_diacritics:
            text = _DIACRITICS.sub("", text)

        if self.remove_tatweel:
            text = _TATWEEL.sub("", text)

        if self.normalize_alef:
            text = text.translate(_ALEF_MAP)

        if self.normalize_ya:
            text = text.translate(_YA_MAP)

        if self.normalize_ta_marbuta:
            text = text.translate(_TA_MARBUTA_MAP)

        return text