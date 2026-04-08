"""
OSI News Automation System - Batch Scraper
===========================================
Scrapes articles from multiple configured news sources.
Supports both web scraping and RSS feed methods.

Uses an async pipeline internally (trafilatura + curl_cffi + tenacity)
while preserving the original synchronous public interface so callers
like run_automation.py need zero changes.
"""

import asyncio
import yaml
from bs4 import BeautifulSoup
import requests
from loguru import logger
from typing import List, Dict, Optional
import time
from random import uniform
import os
import sys
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import logging as _std_logging

import trafilatura
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# curl_cffi import guard — may not be available on all deployment targets
try:
    from curl_cffi.requests import AsyncSession as CurlSession
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.scrapers.news_scraper import extract_location, MAJOR_CITIES
from src.scrapers.rss_scraper import parse_rss_feed, parse_rss_feed_async
from src.scrapers.scrapling_fetcher import fetch_with_scrapling
from src.scrapers.guardian_fetcher import fetch_guardian_articles

# One-time log guard for curl_cffi fallback
_CURL_FALLBACK_LOGGED = False


# ===========================================
# URL FILTERING
# ===========================================

# URL patterns to exclude (ads, videos, galleries, etc.)
EXCLUDED_URL_PATTERNS = [
    '/video/', '/videos/', '/gallery/', '/galleries/',
    '/live/', '/sport/', '/sports/',
    '/weather/', '/lottery/', '/games/',
    '/login', '/signup', '/subscribe',
    '/ads/', '/advertisement/',
    '.pdf', '.jpg', '.png', '.gif',
    'facebook.com', 'twitter.com', 'instagram.com',
    '/author/', '/tag/', '/category/',
]


def is_valid_article_url(url: str, source_url: str) -> bool:
    """
    Check if URL is likely a valid news article.
    
    Args:
        url: URL to check.
        source_url: Base URL of the source.
        
    Returns:
        bool: True if URL appears to be a valid article.
    """
    if not url:
        return False
    
    # Must be HTTP(S)
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Must be from same domain (or subdomain)
    source_domain = urlparse(source_url).netloc.replace('www.', '')
    url_domain = urlparse(url).netloc.replace('www.', '')
    
    if source_domain not in url_domain and url_domain not in source_domain:
        return False
    
    # Check excluded patterns
    url_lower = url.lower()
    for pattern in EXCLUDED_URL_PATTERNS:
        if pattern in url_lower:
            return False
    
    # URL should have some path (not just homepage)
    path = urlparse(url).path
    if not path or path == '/':
        return False
    
    return True


def normalize_url(href: str, base_url: str) -> Optional[str]:
    """
    Convert relative URL to absolute URL.
    
    Args:
        href: Raw href from page.
        base_url: Base URL of the page.
        
    Returns:
        Absolute URL or None if invalid.
    """
    if not href:
        return None
    
    # Already absolute
    if href.startswith(('http://', 'https://')):
        return href
    
    # Protocol-relative
    if href.startswith('//'):
        return 'https:' + href
    
    # Relative URL
    return urljoin(base_url, href)


# ===========================================
# URL EXTRACTION
# ===========================================


def _parse_article_urls_from_soup(soup, source: Dict) -> List[str]:
    """
    Extract article URLs from a BeautifulSoup-parsed page.
    Shared helper to avoid duplicating CSS-selector logic.
    """
    article_urls = []
    max_per_source = source.get('max_articles_per_source', 10)

    # Try configured selector first
    if 'selectors' in source and source['selectors'].get('article_url'):
        selector = source['selectors']['article_url']
        links = soup.select(selector)
        for link in links:
            href = link.get('href')
            url = normalize_url(href, source['url'])
            if url and is_valid_article_url(url, source['url']):
                if url not in article_urls:
                    article_urls.append(url)
                    if len(article_urls) >= max_per_source:
                        break

    # Fallback: common article link patterns
    if not article_urls:
        common_selectors = [
            'article a', 'h2 a', 'h3 a',
            '.article a', '.story a', '.news-item a',
            '[data-testid*="headline"] a', '[class*="headline"] a',
        ]
        for selector in common_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    url = normalize_url(href, source['url'])
                    if url and is_valid_article_url(url, source['url']):
                        if url not in article_urls:
                            article_urls.append(url)
                            if len(article_urls) >= max_per_source:
                                break
                if article_urls:
                    break
            except Exception:
                continue

    return article_urls


def _fetch_page_with_curl_cffi(source: Dict) -> List[str]:
    """
    Fallback URL extractor using curl_cffi TLS impersonation.
    Called when the primary requests.get() path fails for any reason.
    """
    source_name = source.get('name', 'Unknown')
    try:
        from curl_cffi.requests import Session as CurlSyncSession
        with CurlSyncSession(impersonate="chrome110") as s:
            timeout = int(os.getenv('REQUEST_TIMEOUT_SECONDS', 30))
            resp = s.get(source['url'], timeout=timeout)
            resp.raise_for_status()
            html = resp.text
            logger.info(f"{source_name}: curl_cffi page fetch succeeded")
            soup = BeautifulSoup(html, 'lxml')
            urls = _parse_article_urls_from_soup(soup, source)
            logger.info(f"Found {len(urls)} article URLs from {source_name} via curl_cffi")
            return urls
    except Exception as e:
        logger.warning(f"{source_name}: curl_cffi page fetch also failed: {e}")
        return []


def extract_article_urls_from_page(source: Dict) -> List[str]:
    """
    Extract article URLs from a news source homepage using CSS selectors.
    
    Args:
        source: Source configuration dictionary from YAML.
        
    Returns:
        List of article URLs.
    """
    source_name = source.get('name', 'Unknown')
    try:
        logger.debug(f"Extracting URLs from: {source['url']}")

        headers = {
            'User-Agent': os.getenv('USER_AGENT', 'RobinOSI-Bot/1.0'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        timeout = int(os.getenv('REQUEST_TIMEOUT_SECONDS', 30))
        response = requests.get(source['url'], headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')
        article_urls = _parse_article_urls_from_soup(soup, source)

        logger.info(f"Found {len(article_urls)} article URLs from {source_name}")
        return article_urls

    except requests.exceptions.Timeout:
        logger.warning(f"{source_name}: timeout, retrying with curl_cffi...")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if hasattr(e, 'response') and e.response is not None else '?'
        logger.warning(f"{source_name}: HTTP {status}, retrying with curl_cffi...")
    except requests.exceptions.RequestException as e:
        logger.warning(f"{source_name}: {type(e).__name__}, retrying with curl_cffi...")
    except Exception as e:
        logger.warning(f"{source_name}: URL extraction error ({type(e).__name__}), retrying with curl_cffi...")

    # ── All-failures curl_cffi fallback ──
    return _fetch_page_with_curl_cffi(source)


def extract_article_urls_from_rss(source: Dict) -> List[str]:
    """
    Extract article URLs from a news source's RSS feed.
    
    Args:
        source: Source configuration dictionary from YAML.
        
    Returns:
        List of article URLs.
    """
    urls = []
    max_per_source = source.get('max_articles_per_source', 10)
    
    if 'rss_feed' not in source:
        return []
    
    rss_url = source['rss_feed']
    entries = parse_rss_feed(rss_url, limit=max_per_source)
    
    for entry in entries:
        if entry.get('link'):
            urls.append(entry['link'])
    
    return urls


def extract_article_urls(source: Dict, prefer_rss: bool = True) -> List[str]:
    """
    Extract article URLs from a source using best available method.
    
    Args:
        source: Source configuration dictionary.
        prefer_rss: If True, try RSS first (more reliable).
        
    Returns:
        List of article URLs.
    """
    urls = []
    
    if prefer_rss and 'rss_feed' in source:
        # Try RSS first (more reliable)
        urls = extract_article_urls_from_rss(source)
        
        if urls:
            logger.debug(f"Using RSS for {source['name']}: {len(urls)} URLs")
            return urls
    
    # Fall back to page scraping
    urls = extract_article_urls_from_page(source)
    
    if urls:
        logger.debug(f"Using page scraping for {source['name']}: {len(urls)} URLs")
    
    return urls


async def extract_article_urls_async(source: Dict, prefer_rss: bool = True) -> List[str]:
    """
    Async version of extract_article_urls.

    Directly awaits parse_rss_feed_async() instead of going through the
    synchronous wrapper, avoiding RuntimeError from nested asyncio.run().

    Args:
        source: Source configuration dictionary.
        prefer_rss: If True, try RSS first.

    Returns:
        List of article URLs.
    """
    urls = []
    source_name = source.get('name', 'Unknown')

    if prefer_rss and 'rss_feed' in source:
        rss_url = source['rss_feed']
        max_per_source = source.get('max_articles_per_source', 10)
        entries = await parse_rss_feed_async(rss_url, limit=max_per_source)

        for entry in entries:
            if entry.get('link'):
                urls.append(entry['link'])

        if urls:
            logger.debug(f"Using RSS for {source_name}: {len(urls)} URLs")
            return urls

    # Tier 2: Fall back to page scraping (requests → curl_cffi fallback)
    urls = await asyncio.to_thread(extract_article_urls_from_page, source)

    if urls:
        logger.debug(f"Using page scraping for {source_name}: {len(urls)} URLs")
        return urls

    # Tier 3: Scrapling homepage fetch as last resort
    try:
        scrapling_html = await fetch_with_scrapling(source['url'])
        if scrapling_html:
            soup = BeautifulSoup(scrapling_html, 'lxml')
            urls = _parse_article_urls_from_soup(soup, source)
            if urls:
                logger.info(
                    f"Using Scrapling homepage fetch for {source_name}: "
                    f"{len(urls)} URLs"
                )
                return urls
    except Exception as e:
        logger.debug(f"Scrapling homepage fetch failed for {source_name}: {e}")

    return urls


# ===========================================
# ASYNC ARTICLE EXTRACTION
# ===========================================

# Standard tenacity logger bridge for before_sleep_log
_tenacity_logger = _std_logging.getLogger("tenacity.retry")
_tenacity_logger.setLevel(_std_logging.WARNING)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_tenacity_logger, _std_logging.WARNING),
    reraise=True,
)
async def _fetch_html(url: str) -> str:
    """
    Fetch raw HTML from a URL using curl_cffi (preferred) or httpx (fallback).

    Wrapped in tenacity retry: 3 attempts, exponential backoff.
    Raises on failure after all retries so the caller can handle it.
    """
    global _CURL_FALLBACK_LOGGED

    headers = {
        "User-Agent": os.getenv("USER_AGENT", "RobinOSI-Bot/1.0"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
    }
    timeout_secs = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 30))

    if _CURL_AVAILABLE:
        async with CurlSession(impersonate="chrome120") as session:
            resp = await session.get(url, headers=headers, timeout=timeout_secs)
            resp.raise_for_status()
            return resp.text
    else:
        if not _CURL_FALLBACK_LOGGED:
            logger.debug("curl_cffi unavailable — falling back to httpx")
            _CURL_FALLBACK_LOGGED = True
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_secs),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text


async def _fetch_and_extract(url: str, source_name: str = "") -> Optional[Dict]:
    """
    Fetch a page and extract article content using a tiered strategy:
      Tier 1: Scrapling anti-bot bypass (raw HTML → trafilatura)
      Tier 2: curl_cffi / httpx fetch → trafilatura

    Returns an article dict matching the legacy schema produced by
    scrape_single_article(), or None on any failure.
    """
    # ── Tier 1: Scrapling — anti-bot bypass (primary) ──
    scrapling_html = await fetch_with_scrapling(url)
    if scrapling_html:
        # Pass the HTML through the existing trafilatura extraction path
        html_content = scrapling_html
    else:
        html_content = None  # existing fetch logic fills this below

    # ── Tier 2: Existing fetch path (curl_cffi / httpx) ──
    if html_content is None:
        try:
            html_content = await _fetch_html(url)
        except Exception as e:
            logger.warning(f"All retries exhausted for {url}: [{type(e).__name__}] {e}")
            return None

    if not html_content:
        logger.warning(f"Empty HTML response from: {url}")
        return None

    # --- Content extraction via trafilatura ---
    story = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
        url=url,
    )

    if not story:
        logger.warning(f"trafilatura returned no content for: {url}")
        return None

    # --- Word count guard ---
    word_count = len(story.split())
    min_words = int(os.getenv("MIN_ARTICLE_WORDS", 50))
    if word_count < min_words:
        logger.warning(
            f"Article too short ({word_count} words, minimum {min_words}): {url}"
        )
        return None

    # --- Metadata extraction ---
    metadata = trafilatura.extract_metadata(html_content, default_url=url)

    heading = ""
    authors = []
    publish_date = datetime.utcnow().isoformat()
    top_image = ""
    meta_description = ""

    if metadata:
        heading = metadata.title or ""
        if metadata.author:
            # trafilatura returns author as semicolon-separated string
            authors = [
                a.strip() for a in metadata.author.split(";") if a.strip()
            ]
        if metadata.date:
            publish_date = str(metadata.date)
        top_image = metadata.image or ""
        meta_description = metadata.description or ""

    if not heading:
        # Last resort: take first line of story
        heading = story.split("\n")[0][:200]

    # --- Language detection ---
    try:
        from langdetect import detect, LangDetectException
        sample_text = story[:500] if len(story) > 500 else story
        language = detect(sample_text)
    except Exception:
        language = "en"

    # --- Location extraction ---
    location = extract_location(story)

    # --- Build the article dict (schema-identical to news_scraper output) ---
    article_data = {
        "heading": heading,
        "story": story,
        "source_url": url,
        "source_name": source_name,
        "authors": authors,
        "publish_date": publish_date,
        "top_image": top_image,
        "location": location,
        "language": language,
        "scraped_at": datetime.utcnow().isoformat(),
        "word_count": word_count,
        "meta_keywords": [],
        "meta_description": meta_description,
    }

    logger.info(f"✅ Scraped: {heading[:50]}... ({word_count} words)")
    return article_data


# ===========================================
# BATCH SCRAPING
# ===========================================

def load_news_sources(
    config_path: str = 'config/news_sources.yaml',
    run_number: int = None
) -> List[Dict]:
    """
    Load enabled news sources, applying tier-based scheduling.

    Tier logic:
      - Priority 1-2 (core sources): included every run
      - Priority 3   (regional):     included every other run (even run numbers)
      - Priority 4+  (specialist):   included every 3rd run

    Args:
        config_path: Path to news_sources.yaml.
        run_number:  Monotonic run counter from DB or env. If None, all sources load
                     (safe default for dry-runs and tests).

    Returns:
        List of enabled, scheduled source configurations sorted by priority.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        all_sources = [s for s in config.get('sources', []) if s.get('enabled', True)]

        if run_number is None:
            selected = all_sources
        else:
            selected = []
            for source in all_sources:
                priority = source.get('priority', 5)
                if priority <= 2:
                    selected.append(source)
                elif priority == 3:
                    if run_number % 2 == 0:
                        selected.append(source)
                else:
                    if run_number % 3 == 0:
                        selected.append(source)

        selected.sort(key=lambda x: x.get('priority', 5))

        logger.info(
            f"Loaded {len(selected)}/{len(all_sources)} sources "
            f"(run_number={run_number}, scheduling {'active' if run_number is not None else 'disabled'})"
        )
        return selected

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return []


async def _scrape_news_batch_async(
    max_articles: int,
    sources: List[Dict],
    prefer_rss: bool,
    min_per_source: int,
    max_per_source: int,
    session_id: str,
    run_number: int,
) -> List[Dict]:
    """
    Internal async implementation of the batch scrape loop.

    Fetches articles concurrently within each source using asyncio.gather()
    and a shared Semaphore(10) to cap total in-flight requests.
    """
    # CONSTRAINT 3: create semaphore inside the async function,
    # not at module level, so it's bound to the current event loop.
    sem = asyncio.Semaphore(10)

    articles: List[Dict] = []
    failed_urls: List[str] = []
    sources_scraped = 0
    start_time = time.time()
    source_stats = defaultdict(lambda: {"attempted": 0, "succeeded": 0, "failed": 0})

    logger.info(f"🚀 Starting batch scrape from {len(sources)} sources...")
    logger.info(f"   Target: {max_articles} articles | Session: {session_id}")

    # --- Fix 1: Pre-fetch all RSS feeds concurrently ---
    url_tasks = [
        extract_article_urls_async(source, prefer_rss=prefer_rss)
        for source in sources
    ]
    url_results = await asyncio.gather(*url_tasks, return_exceptions=True)
    prefetched_urls: Dict[str, List[str]] = {}
    for source, result in zip(sources, url_results):
        sname = source.get('name', 'Unknown')
        if isinstance(result, Exception):
            logger.warning(f"Pre-fetch failed for {sname}: {result}")
            prefetched_urls[sname] = []
        else:
            prefetched_urls[sname] = result
    # --- END Fix 1 pre-fetch ---

    async def _bounded_fetch(url, source_name, rate_limit):
        async with sem:
            await asyncio.sleep(rate_limit)
            try:
                return await _fetch_and_extract(url, source_name)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: [{type(e).__name__}] {e}")
                return None

    for source in sources:
        source_name = source.get('name', 'Unknown')
        logger.info(f"\n📰 Scraping: {source_name} (Priority: {source.get('priority', 5)})")

        try:
            # Get article URLs from pre-fetched results
            article_urls = prefetched_urls.get(source_name, [])

            if not article_urls:
                logger.warning(f"   No articles found from {source_name}")
                continue

            # Respect per-source limit from YAML (max_articles_per_source)
            # but do NOT apply a global budget cap — scrape everything available.
            per_source_cap = source.get('max_articles_per_source', max_per_source)
            article_urls = article_urls[:per_source_cap]
            logger.info(f"   Found {len(article_urls)} article URLs")

            rate_limit = source.get('rate_limit_delay', 2)

            # Launch concurrent fetches for this source
            tasks = [
                _bounded_fetch(url, source_name, rate_limit)
                for url in article_urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            source_articles = 0
            for i, result in enumerate(results):
                source_stats[source_name]["attempted"] += 1
                if result is not None:
                    result['source_region'] = source.get('region', 'Unknown')
                    result['session_id'] = session_id
                    result['priority'] = source.get('priority', 5)
                    articles.append(result)
                    source_articles += 1
                    source_stats[source_name]["succeeded"] += 1
                    logger.info(f"   ✅ [{len(articles)}/{max_articles}] {result['heading'][:50]}...")
                else:
                    if i < len(article_urls):
                        failed_urls.append(article_urls[i])
                    source_stats[source_name]["failed"] += 1

            if source_articles > 0:
                sources_scraped += 1
                logger.info(f"   Scraped {source_articles} articles from {source_name}")
            else:
                attempted = source_stats[source_name]["attempted"]
                if attempted > 0:
                    logger.warning(
                        f"   ⚠️  ZERO yield from {source_name} "
                        f"({attempted} URLs attempted, 0 succeeded). "
                        f"Check RSS feed or geoblocking."
                    )

        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
            continue

    # Calculate duration
    duration = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Batch Scrape Summary")
    logger.info("=" * 60)
    logger.info(f"   Articles scraped: {len(articles)}")
    logger.info(f"   Sources used: {sources_scraped}/{len(sources)}")
    logger.info(f"   Failed URLs: {len(failed_urls)}")
    logger.info(f"   Duration: {duration:.1f} seconds")
    logger.info(f"   Session ID: {session_id}")
    logger.info("=" * 60 + "\n")

    dead_sources = [
        name for name, s in source_stats.items()
        if s["attempted"] > 0 and s["succeeded"] == 0
    ]
    low_yield_sources = [
        name for name, s in source_stats.items()
        if s["attempted"] > 0 and 0 < s["succeeded"] < 2
    ]

    if dead_sources:
        logger.warning(f"   🚨 Dead sources  : {', '.join(dead_sources)}")
    if low_yield_sources:
        logger.warning(f"   ⚠️  Low yield     : {', '.join(low_yield_sources)}")

    _persist_source_health(session_id, source_stats)

    return articles


def scrape_news_batch(
    max_articles: int = 9999,
    sources: List[Dict] = None,
    prefer_rss: bool = True,
    min_per_source: int = 2,
    max_per_source: int = int(os.getenv('PER_SOURCE_ARTICLE_LIMIT', 10)),
    session_id: str = None,
    run_number: int = None
) -> List[Dict]:
    """
    Scrape articles from multiple news sources.
    
    Loads configured sources, extracts article URLs, and scrapes each article
    until max_articles is reached. Distributes across sources by priority.
    
    Args:
        max_articles: Maximum total articles to scrape.
        sources: List of source configs. If None, loads from YAML.
        prefer_rss: Prefer RSS feeds over page scraping.
        min_per_source: Minimum articles to try from each source.
        max_per_source: Maximum articles per source.
        session_id: Optional session ID to attach to articles.
        run_number: Optional run counter for tier-based scheduling.
        
    Returns:
        List of successfully scraped article dictionaries.
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = f"BATCH_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Load sources if not provided
    if sources is None:
        sources = load_news_sources(run_number=run_number)
    
    if not sources:
        logger.error("No news sources configured")
        return []
    
    return asyncio.run(
        _scrape_news_batch_async(
            max_articles, sources, prefer_rss,
            min_per_source, max_per_source,
            session_id, run_number
        )
    )


def scrape_specific_sources(
    source_names: List[str],
    max_articles: int = 50,
    prefer_rss: bool = True
) -> List[Dict]:
    """
    Scrape articles from specific named sources.
    
    Args:
        source_names: List of source names to scrape.
        max_articles: Maximum total articles.
        prefer_rss: Prefer RSS feeds.
        
    Returns:
        List of scraped articles.
    """
    all_sources = load_news_sources()
    
    # Filter to requested sources
    selected_sources = [
        s for s in all_sources 
        if s.get('name', '') in source_names
    ]
    
    if not selected_sources:
        logger.warning(f"None of the requested sources found: {source_names}")
        return []
    
    return scrape_news_batch(
        max_articles=max_articles,
        sources=selected_sources,
        prefer_rss=prefer_rss
    )


def scrape_by_tier(
    tiers: List[int],
    max_articles: int = 50,
    prefer_rss: bool = True,
    session_id: str = None,
) -> List[Dict]:
    """
    Scrape only sources matching the given priority tiers.

    Example:
        # Fast run — tier 1 wire services only
        articles = scrape_by_tier([1], max_articles=20)

        # Full run — all tiers
        articles = scrape_by_tier([1, 2, 3, 4], max_articles=50)

    Args:
        tiers:        List of priority numbers to include.
        max_articles: Article cap.
        prefer_rss:   Prefer RSS over page scraping.
        session_id:   Optional session ID.

    Returns:
        List of scraped article dictionaries.
    """
    all_sources = load_news_sources()   # no run_number = load all enabled
    selected = [s for s in all_sources if s.get('priority', 5) in tiers]

    if not selected:
        logger.warning(f"No sources found for tiers: {tiers}")
        return []

    logger.info(f"Tier-filtered scrape: tiers={tiers}, sources={len(selected)}")
    return scrape_news_batch(
        max_articles=max_articles,
        sources=selected,
        prefer_rss=prefer_rss,
        session_id=session_id,
    )


def _persist_source_health(session_id: str, source_stats: dict) -> None:
    """
    Write per-source yield data to MongoDB for dashboard visibility.
    Failures accumulate as a counter — the dashboard flags sources that
    have been dead for N consecutive runs.
    """
    try:
        from src.database.mongo_client import get_client
        db = get_client()
        if not db._ensure_connected():
            return

        now = datetime.utcnow()
        for source_name, stats in source_stats.items():
            yield_rate = (
                stats["succeeded"] / stats["attempted"]
                if stats["attempted"] > 0 else None
            )
            db.db["source_health"].update_one(
                {"source_name": source_name},
                {
                    "$set": {
                        "last_seen": now,
                        "last_yield_rate": yield_rate,
                        "last_session_id": session_id,
                    },
                    "$push": {
                        "recent_yields": {
                            "$each": [{"ts": now, "rate": yield_rate}],
                            "$slice": -20,
                        }
                    },
                    "$inc": {
                        "consecutive_failures": 0 if (yield_rate or 0) > 0 else 1,
                        "total_runs": 1,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            if (yield_rate or 0) > 0:
                db.db["source_health"].update_one(
                    {"source_name": source_name},
                    {"$set": {"consecutive_failures": 0}}
                )

    except Exception as e:
        logger.warning(f"Could not persist source health: {e}")


# ===========================================
# TESTING
# ===========================================

def test_batch_scraper():
    """Test batch scraping with a small number of articles."""
    print("\n" + "="*60)
    print("🧪 Batch Scraper Test")
    print("="*60)
    
    # Test with just 5 articles
    articles = scrape_news_batch(max_articles=5, prefer_rss=True)
    
    print(f"\n📰 Scraped {len(articles)} articles:")
    print("-" * 40)
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['heading'][:55]}...")
        print(f"   Source: {article.get('source_name', 'Unknown')}")
        print(f"   Words: {article.get('word_count', 0)}")
        print()
    
    print("="*60)
    if len(articles) > 0:
        print("✅ Batch scraper test passed!")
    else:
        print("⚠️ No articles scraped (network or source issues)")
    print("="*60 + "\n")
    
    return articles


if __name__ == "__main__":
    test_batch_scraper()
