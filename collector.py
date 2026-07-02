"""
collect_labeled_data.py  (Arabic news, large-scale version)

Collects 20,000+ Arabic news articles for propaganda/statement/attribution
labeling. RSS feeds only expose the last ~50-100 items, so to reach
five-figure volumes this crawls SITEMAPS (which list thousands of historical
article URLs) across multiple Arabic news sources, fetches each article
concurrently, and stages them as unlabeled JSONL records.

Workflow:
  1. FETCH-CONFIG - crawl sitemaps for a list of Arabic sources (see
                     ARABIC_SOURCES below / sources.json), pull article
                     text + metadata, stage as unlabeled records.
                     Supports --target (default 20000) and resumes
                     automatically across runs (dedupes by URL).
  2. ANNOTATE     - interactively attach propaganda_label / statement_type /
                     attribution_label / verified / reliability_score to
                     staged records (these need a human judgment call).

Usage:
    # crawl the built-in Arabic sources list toward 20,000 records
    python collect_labeled_data.py fetch-config --out staged.jsonl --target 20000

    # or use your own sources.json (see ARABIC_SOURCES format below)
    python collect_labeled_data.py fetch-config --config sources.json --out staged.jsonl --target 20000

    # single source only
    python collect_labeled_data.py fetch \
        --sitemap-url https://www.dw.com/ar/sitemap.xml \
        --source DW --source-bias center --region Germany \
        --out staged.jsonl --target 5000

    python collect_labeled_data.py annotate --in staged.jsonl --out labeled.jsonl
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date



PROPAGANDA_MAP = {
    "neutral": 0, "loaded_language": 1, "propaganda": 2,
    "sensationalism": 3, "false_dichotomy": 4, "fear_appeal": 5,
    "doubt_casting": 6, "exaggeration": 7, "stereotyping": 8
}
STATEMENT_MAP = {
    "fact": 0, "opinion": 1, "speculation": 2, "reporting": 3
}
ATTRIBUTION_MAP = {
    "supported_claim": 0, "unsupported_claim": 1, "quote_present": 2, "direct_source": 3
}

REQUIRED_FIELDS = [
    "text", "source", "source_bias", "region",
    "propaganda_label", "attribution_label", "statement_type",
    "verified", "reliability_score", "title", "date"
]

MIN_TEXT_CHARS = 200 


ARABIC_SOURCES = [
    {"sitemap_url": "https://www.dw.com/ar/sitemap.xml",            "source": "DW Arabic",       "source_bias": "center", "region": "Germany"},
    {"sitemap_url": "https://www.aljazeera.net/sitemap.xml",        "source": "Al Jazeera",       "source_bias": "center", "region": "Qatar"},
    {"sitemap_url": "https://www.bbc.com/arabic/sitemap.xml",       "source": "BBC Arabic",       "source_bias": "center", "region": "UK"},
    {"sitemap_url": "https://arabic.cnn.com/sitemap.xml",           "source": "CNN Arabic",       "source_bias": "center", "region": "USA"},
    {"sitemap_url": "https://www.skynewsarabia.com/sitemap.xml",    "source": "Sky News Arabia",  "source_bias": "center", "region": "UAE"},
    {"sitemap_url": "https://www.france24.com/ar/sitemap.xml",      "source": "France24 Arabic",  "source_bias": "center", "region": "France"},
    {"sitemap_url": "https://www.independentarabia.com/sitemap.xml","source": "Independent Arabia","source_bias": "center", "region": "UK"},
    {"sitemap_url": "https://www.alarabiya.net/sitemap.xml",        "source": "Al Arabiya",       "source_bias": "center", "region": "Saudi Arabia"},
]

_write_lock = threading.Lock()


def load_jsonl(path):
    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def append_jsonl(path, record):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_record(record):
    missing = [k for k in REQUIRED_FIELDS if k not in record]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    return True


def looks_arabic(text, threshold=0.3):
    if not text:
        return False
    arabic_chars = sum(1 for ch in text if '\u0600' <= ch <= '\u06FF')
    return (arabic_chars / max(len(text), 1)) >= threshold




def discover_urls_from_sitemap(sitemap_url, requests, seen_sitemaps=None, max_urls=100000):
    """Recursively walk sitemap indexes and return a flat list of article URLs."""
    import xml.etree.ElementTree as ET

    if seen_sitemaps is None:
        seen_sitemaps = set()
    if sitemap_url in seen_sitemaps:
        return []
    seen_sitemaps.add(sitemap_url)

    try:
        resp = requests.get(sitemap_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"  [warn] could not parse sitemap {sitemap_url}: {e}")
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []

    sitemap_tags = root.findall("sm:sitemap/sm:loc", ns)
    if sitemap_tags:
        for loc in sitemap_tags:
            child_urls = discover_urls_from_sitemap(loc.text.strip(), requests, seen_sitemaps, max_urls)
            urls.extend(child_urls)
            if len(urls) >= max_urls:
                return urls[:max_urls]
    else:
        for loc in root.findall("sm:url/sm:loc", ns):
            if loc.text:
                urls.append(loc.text.strip())

    return urls[:max_urls]



def fetch_article(url, requests, BeautifulSoup):
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("meta", property="og:title") or soup.find("title")
    title = title_tag.get("content") if title_tag and title_tag.has_attr("content") else (
        title_tag.get_text(strip=True) if title_tag else ""
    )

    date_tag = soup.find("meta", property="article:published_time")
    pub_date = date_tag.get("content")[:10] if date_tag and date_tag.has_attr("content") else str(date.today())

    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if p).strip()

    return title, text, pub_date


def worker(url, source, source_bias, region, out_path, existing_urls, requests, BeautifulSoup, counter, target, lock):
    if url in existing_urls:
        return False
    try:
        title, text, pub_date = fetch_article(url, requests, BeautifulSoup)
    except Exception:
        return False

    if len(text) < MIN_TEXT_CHARS or not looks_arabic(text):
        return False

    record = {
        "text": text,
        "source": source,
        "source_bias": source_bias,
        "region": region,
        "propaganda_label": None,
        "attribution_label": None,
        "statement_type": None,
        "verified": False,
        "reliability_score": None,
        "title": title,
        "date": pub_date,
        "_source_url": url,
    }
    append_jsonl(out_path, record)
    existing_urls.add(url)

    with lock:
        counter[0] += 1
        if counter[0] % 100 == 0:
            print(f"  ... {counter[0]} records collected (target {target})")

    return True




def fetch_source(sitemap_url, source, source_bias, region, out_path, target,
                  max_workers=10, counter=None, lock=None):
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("Missing deps. Install with:\n  pip install requests beautifulsoup4 --break-system-packages")
        sys.exit(1)

    if counter is None:
        counter = [0]
    if lock is None:
        lock = threading.Lock()

    existing = load_jsonl(out_path)
    existing_urls = {r.get("_source_url") for r in existing if r.get("_source_url")}
    counter[0] = len([r for r in existing if r.get("_source_url")])

    print(f"[{source}] discovering URLs from sitemap: {sitemap_url}")
    urls = discover_urls_from_sitemap(sitemap_url, requests, max_urls=max(target * 3, 3000))
    print(f"[{source}] found {len(urls)} candidate URLs, need {max(target - counter[0], 0)} more records")

    if counter[0] >= target:
        print(f"[{source}] target already met ({counter[0]}/{target}).")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for url in urls:
            if counter[0] >= target:
                break
            if url in existing_urls:
                continue
            futures.append(pool.submit(
                worker, url, source, source_bias, region, out_path,
                existing_urls, requests, BeautifulSoup, counter, target, lock
            ))

        for f in as_completed(futures):
            if counter[0] >= target:
                break
            f.result()

    print(f"[{source}] done. total records so far: {counter[0]}")




def fetch_from_config(out_path, target, config_path=None, max_workers=10):
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            sources = json.load(f)
    else:
        sources = ARABIC_SOURCES
        print(f"No --config given, using {len(sources)} built-in Arabic sources.")

    if not sources:
        print("No sources to crawl.")
        return

    per_source_target = max(1, target // len(sources))
    counter = [len(load_jsonl(out_path))]
    lock = threading.Lock()

    for src in sources:
        if counter[0] >= target:
            break
        remaining = target - counter[0]
        this_target = min(per_source_target, remaining) + counter[0]
        fetch_source(
            src["sitemap_url"], src["source"], src["source_bias"], src["region"],
            out_path, this_target, max_workers=max_workers, counter=counter, lock=lock
        )

    print(f"\nGrand total: {counter[0]} records in {out_path}")
    if counter[0] < target:
        print(f"Short of target ({target}). Add more sources / raise per-source caps "
              f"and re-run fetch-config again -- it resumes automatically (dedupes by URL).")



def prompt_choice(prompt, choices_map):
    choices = list(choices_map.keys())
    while True:
        print(f"\n{prompt}")
        for i, c in enumerate(choices):
            print(f"  [{i}] {c}")
        raw = input("choice #: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(choices):
            return choices[int(raw)]
        print("Invalid choice, try again.")


def annotate(in_path, out_path):
    staged = load_jsonl(in_path)
    done_urls = {r.get("_source_url") for r in load_jsonl(out_path) if r.get("_source_url")}

    pending = [r for r in staged if r.get("_source_url") not in done_urls]
    if not pending:
        print("Nothing left to annotate.")
        return

    print(f"{len(pending)} record(s) pending annotation.")
    for i, record in enumerate(pending):
        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(pending)}] {record.get('title')}")
        print("-" * 80)
        print(record["text"][:800] + ("..." if len(record["text"]) > 800 else ""))
        print("-" * 80)

        record["propaganda_label"] = prompt_choice("Propaganda label:", PROPAGANDA_MAP)
        record["statement_type"] = prompt_choice("Statement type:", STATEMENT_MAP)
        record["attribution_label"] = prompt_choice("Attribution label:", ATTRIBUTION_MAP)

        verified_raw = input("Verified? (y/n) [n]: ").strip().lower()
        record["verified"] = verified_raw == "y"

        score_raw = input("Reliability score (0.0-1.0) [0.5]: ").strip()
        try:
            record["reliability_score"] = float(score_raw) if score_raw else 0.5
        except ValueError:
            record["reliability_score"] = 0.5

        validate_record({k: v for k, v in record.items() if k != "_source_url"})
        append_jsonl(out_path, record)
        print("saved.")



def main():
    parser = argparse.ArgumentParser(description="Collect (20k+) and label Arabic news for propaganda analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Crawl one source's sitemap for articles")
    p_fetch.add_argument("--sitemap-url", required=True)
    p_fetch.add_argument("--source", required=True)
    p_fetch.add_argument("--source-bias", required=True, choices=["left", "center", "right"])
    p_fetch.add_argument("--region", required=True)
    p_fetch.add_argument("--out", default="staged.jsonl")
    p_fetch.add_argument("--target", type=int, default=20000)
    p_fetch.add_argument("--max-workers", type=int, default=10)

    p_fetch_cfg = sub.add_parser("fetch-config", help="Crawl multiple Arabic sources toward a shared target")
    p_fetch_cfg.add_argument("--config", default=None, help="Optional JSON file; defaults to built-in ARABIC_SOURCES")
    p_fetch_cfg.add_argument("--out", default="staged.jsonl")
    p_fetch_cfg.add_argument("--target", type=int, default=20000)
    p_fetch_cfg.add_argument("--max-workers", type=int, default=10)

    p_annotate = sub.add_parser("annotate", help="Interactively label staged articles")
    p_annotate.add_argument("--in", dest="in_path", default="staged.jsonl")
    p_annotate.add_argument("--out", dest="out_path", default="labeled.jsonl")

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_source(args.sitemap_url, args.source, args.source_bias, args.region,
                      args.out, args.target, max_workers=args.max_workers)
    elif args.command == "fetch-config":
        fetch_from_config(args.out, args.target, config_path=args.config, max_workers=args.max_workers)
    elif args.command == "annotate":
        annotate(args.in_path, args.out_path)


if __name__ == "__main__":
    main()