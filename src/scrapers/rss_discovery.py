"""
src/scrapers/rss_discovery.py
=============================
Automatically discovers and validates RSS feed URLs.
Run this to fix dead feeds in news_sources.yaml.

Usage:
    python src/scrapers/rss_discovery.py [--dry-run] [--source NAME]

Key fix vs original design: uses regex line-replace instead of yaml.dump
so that comments, ordering, and formatting in news_sources.yaml are preserved.
"""

import asyncio
import re
import httpx
import feedparser
import yaml
from pathlib import Path
from typing import Optional
from loguru import logger
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# trafilatura is already in requirements
from trafilatura import feeds as trafilatura_feeds


# ── Common URL patterns to brute-force ─────────────────────────────
RSS_PATTERNS = [
    "/feed", "/feed/", "/rss", "/rss/", "/atom", "/atom/",
    "/rss.xml", "/atom.xml", "/feed.xml", "/feed.rss",
    "/index.xml", "/index.rss",
    "/news/rss", "/news/feed", "/news.rss",
    "/latest/rss", "/rss/news", "/en/rss",
    "/en/news/rss.xml", "/?format=rss",
    "/feeds/latest", "/feeds/news.rss",
    "/api/feed", "/feed/news", "/rss/all",
    "/rssfeed/topNews",
    "/xml/rss/all.xml",
    "/rss/home",
    "/rss/topstories",
    "/arcio/rss",
    "/arcio/rss/",
    "/arc/outboundfeeds/rss/",
    "/sito/notizie/mondo/mondo_rss.xml",  # ANSA pattern
]


async def check_feed_url(client: httpx.AsyncClient, url: str) -> bool:
    """Return True if URL is a valid, non-empty RSS/Atom feed."""
    try:
        resp = await client.get(url, timeout=8.0)
        if resp.status_code != 200:
            return False
        parsed = feedparser.parse(resp.text)
        return bool(parsed.entries) and bool(parsed.version)
    except Exception:
        return False


async def discover_via_html(client: httpx.AsyncClient, base_url: str) -> list:
    """Extract RSS URLs from <link rel='alternate'> tags in homepage HTML."""
    found = []
    try:
        resp = await client.get(base_url, timeout=10.0)
        soup = BeautifulSoup(resp.text, "lxml")
        parsed_base = urlparse(base_url)
        for link in soup.find_all("link", rel="alternate"):
            link_type = link.get("type", "")
            href = link.get("href", "")
            if any(ft in link_type for ft in ["rss", "atom", "feed"]):
                if href.startswith("/"):
                    href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                if href:
                    found.append(href)
    except Exception:
        pass
    return found


async def discover_rss_for_source(
    source: dict,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[str]:
    """
    Try all discovery methods for a single source.
    Returns the first working RSS feed URL, or None.
    """
    site_url = source.get("url", "")
    source_name = source.get("name", "Unknown")
    current_feed = source.get("rss_feed", "")

    if not site_url:
        return None

    base = site_url.rstrip("/")
    logger.info(f"  🔍 Discovering RSS for: {source_name}")

    async def _run(c):
        # Method 1: minor URL tweaks on current feed
        if current_feed:
            variants = {current_feed.rstrip("/"), current_feed.rstrip("/") + "/"}
            variants.discard(current_feed)
            for cand in variants:
                if await check_feed_url(c, cand):
                    logger.success(f"    ✅ Minor tweak worked: {cand}")
                    return cand

        # Method 2: trafilatura autodiscovery
        try:
            discovered = trafilatura_feeds.find_feed_urls(site_url)
            for url in (discovered or []):
                if await check_feed_url(c, url):
                    logger.success(f"    ✅ trafilatura found: {url}")
                    return url
        except Exception as e:
            logger.debug(f"    trafilatura failed: {e}")

        # Method 3: HTML <link rel=alternate> tags
        html_feeds = await discover_via_html(c, site_url)
        for url in html_feeds:
            if await check_feed_url(c, url):
                logger.success(f"    ✅ HTML tag found: {url}")
                return url

        # Method 4: brute-force common patterns (all concurrent)
        pattern_urls = [base + p for p in RSS_PATTERNS]
        checks = await asyncio.gather(
            *[check_feed_url(c, u) for u in pattern_urls],
            return_exceptions=True,
        )
        for url, ok in zip(pattern_urls, checks):
            if ok is True:
                logger.success(f"    ✅ Pattern match: {url}")
                return url

        logger.warning(f"    ❌ No RSS found for: {source_name}")
        return None

    if client:
        return await _run(client)
    else:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (RSS Discovery Bot)"},
        ) as c:
            return await _run(c)


def _safe_yaml_update(config_path: Path, name: str, new_feed_url: str) -> bool:
    """
    Replace the rss_feed line for a named source using regex.
    Preserves ALL comments, ordering, and formatting in the YAML.
    Returns True if the file was modified.
    """
    text = config_path.read_text()

    # Build a pattern that matches the source block:
    # Find "- name: <name>" then the next rss_feed: line in that block
    # We use a two-pass approach: locate the name, then patch its rss_feed
    name_escaped = re.escape(name)
    pattern = (
        r"(- name: " + name_escaped + r"(?:\n|.)*?)"
        r"(    rss_feed: )([^\n]+)"
    )

    def replacer(m):
        return m.group(1) + m.group(2) + new_feed_url + "  # auto-fixed by rss_discovery.py"

    new_text, count = re.subn(pattern, replacer, text, count=1)
    if count == 0:
        logger.warning(f"    Could not locate rss_feed line for '{name}' in YAML")
        return False

    config_path.write_text(new_text)
    return True


async def audit_and_fix_yaml(
    config_path: str = "config/news_sources.yaml",
    dry_run: bool = False,
    only_dead: bool = True,
) -> dict:
    """
    Audit all enabled sources in news_sources.yaml.
    For dead feeds, attempt discovery of working replacements.
    Writes fixes back using safe line-replacement (preserves comments).

    Args:
        config_path: path to news_sources.yaml
        dry_run: if True, find fixes but do not write to disk
        only_dead: if True, skip sources whose current feed already works

    Returns summary dict.
    """
    config_file = Path(config_path)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    sources = config.get("sources", [])
    enabled = [s for s in sources if s.get("enabled", True)]
    logger.info(f"Auditing {len(enabled)} enabled sources...")

    fixed = []
    already_working = []
    failed = []

    headers = {"User-Agent": "Mozilla/5.0 (RSS Discovery Bot)"}

    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        for source in enabled:
            name = source.get("name", "Unknown")
            current_feed = source.get("rss_feed", "")

            # Check if current feed is already working
            if current_feed and only_dead:
                if await check_feed_url(client, current_feed):
                    logger.debug(f"  ✅ Already working: {name}")
                    already_working.append(name)
                    continue

            logger.warning(f"  ⚠️  Dead/missing feed: {name} → {current_feed or '(none)'}")

            new_feed = await discover_rss_for_source(source, client=client)

            if new_feed and new_feed != current_feed:
                fixed.append({"name": name, "old_url": current_feed, "new_url": new_feed})
                if not dry_run:
                    if _safe_yaml_update(config_file, name, new_feed):
                        logger.success(f"  💾 Updated YAML for: {name}")
            else:
                failed.append({"name": name, "url": source.get("url", "")})

    # ── Summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RSS DISCOVERY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Already working : {len(already_working)}")
    logger.info(f"🔧 Fixed           : {len(fixed)}")
    for fix in fixed:
        logger.info(f"   {fix['name']}:")
        logger.info(f"     Old: {fix['old_url']}")
        logger.info(f"     New: {fix['new_url']}")
    suffix = " (dry run — no changes written)" if dry_run else ""
    logger.info(f"❌ Could not fix   : {len(failed)}{suffix}")
    for fail in failed:
        logger.info(f"   {fail['name']} — {fail['url']}")
    logger.info("=" * 60)

    return {"already_working": already_working, "fixed": fixed, "failed": failed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover and fix dead RSS feeds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Find fixes but do not write to YAML")
    parser.add_argument("--source", type=str,
                        help="Test a single source by name (e.g. 'Reuters')")
    parser.add_argument("--all", action="store_true",
                        help="Re-check all sources, not just dead ones")
    args = parser.parse_args()

    if args.source:
        async def _test_single():
            result = await discover_rss_for_source(
                {"name": args.source, "url": f"https://www.{args.source.lower().replace(' ', '')}.com", "rss_feed": ""}
            )
            print(f"Found: {result}")
        asyncio.run(_test_single())
    else:
        asyncio.run(audit_and_fix_yaml(
            dry_run=args.dry_run,
            only_dead=not args.all,
        ))
