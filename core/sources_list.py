from typing import List

from core.config import NewsSourceConfig


ARABIC_NEWS_SOURCES: List[NewsSourceConfig] = [
    NewsSourceConfig(
        name="Al Jazeera Arabic",
        name_ar="الجزيرة",
        rss_urls=[
            "https://www.aljazeera.net/aljazeerarss/all.xml",
            "https://www.aljazeera.net/aljazeerarss/a-f/politics.xml",
        ],
        base_url="https://www.aljazeera.net",
        reliability_score=0.78,
        political_lean="center-left",
        region="Qatar",
    ),
    NewsSourceConfig(
        name="Al Arabiya",
        name_ar="العربية",
        rss_urls=["https://www.alarabiya.net/.mrss/ar/last-24-hours.xml"],
        base_url="https://www.alarabiya.net",
        reliability_score=0.72,
        political_lean="center-right",
        region="Saudi Arabia",
    ),
    NewsSourceConfig(
        name="RT Arabic", 
        name_ar="آر تي عربي",
        rss_urls=["https://arabic.rt.com/rss/"],
        base_url="https://arabic.rt.com",
        reliability_score=0.45,
        political_lean="state",
        region="Russia",
    ),
    NewsSourceConfig(
        name="BBC Arabic",
        name_ar="بي بي سي عربي",
        rss_urls=["https://www.bbc.com/arabic/index.xml"], 
        base_url="https://www.bbc.com/arabic",
        reliability_score=0.90,
        political_lean="center",
        region="UK",
    ),
    NewsSourceConfig(
        name="CNN Arabic",
        name_ar="سي إن إن عربي",
        rss_urls=["https://arabic.cnn.com/api/v1/rss/rss.xml"], 
        base_url="https://arabic.cnn.com/",
        reliability_score=0.80,
        political_lean="lean-left",
        region="US",
    ),
    NewsSourceConfig(
        name="DW Arabic",
        name_ar="DW عربي",
        rss_urls=["https://rss.dw.com/atom/rss-ar-all"], 
        base_url="https://www.dw.com/ar",
        reliability_score=0.60,
        political_lean="center",
        region="Germany",
    ),
    NewsSourceConfig(
        name="France 24",
        name_ar="فرانس 24",
        rss_urls=["https://www.france24.com/ar/%D8%A7%D9%84%D8%B4%D8%B1%D9%82-%D8%A7%D9%84%D8%A3%D9%88%D8%B3%D8%B7/rss", #middle east
                  "https://www.france24.com/ar/%D8%A3%D9%88%D8%B1%D9%88%D8%A8%D8%A7/rss", #europe
                  "https://www.france24.com/ar/%D8%A3%D9%85%D8%B1%D9%8A%D9%83%D8%A7/rss", #america
                  "https://www.france24.com/ar/%D8%A2%D8%B3%D9%8A%D8%A7/rss"], #asia
        base_url="https://www.france24.com/ar/",
        reliability_score=0.60,
        political_lean="center-left",
        region="France",
    ),
    NewsSourceConfig(
        name="Sky News Arabia",
        name_ar="سكاي نيوز عربية",
        rss_urls=["https://www.skynewsarabia.com/rss"],
        base_url="https://www.skynewsarabia.com/",
        reliability_score=0.40,
        political_lean="lean-left",
        region="UAE",
    ),
    NewsSourceConfig(
        name="Syria Direct",
        name_ar="سوريا على طول",
        rss_urls=["https://syriadirect.org/ar/feed/"], 
        base_url="https://syriadirect.org/ar/",
        reliability_score=0.60,
        political_lean="left",
        region="Syria",
    ),
    NewsSourceConfig(
        name="Syria TV",
        name_ar="تلفزيون سوريا",
        rss_urls=["https://www.syria.tv/rss"], 
        base_url="https://www.syria.tv/",
        reliability_score=0.60,
        political_lean="center",
        region="Syria",
    ),
    NewsSourceConfig(
        name="Sana",
        name_ar="سانا",
        rss_urls=["https://sana.sy/feed/"], 
        base_url="https://sana.sy/",
        reliability_score=0.70,
        political_lean="state",
        region="Syria",
    ),
    NewsSourceConfig(
        name="Syrian Network for Human Rights",
        name_ar="الشبكة السورية لحقوق الإنسان",
        rss_urls=["https://snhr.org/arabic/feed/"], 
        base_url="https://snhr.org/arabic/",
        reliability_score=0.90,
        political_lean="center",
        region="Syria",
    ),
]