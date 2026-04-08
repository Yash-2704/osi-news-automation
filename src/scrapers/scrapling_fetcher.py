# src/scrapers/scrapling_fetcher.py

"""
Scrapling-based fetcher for anti-bot protected sources.

Deploy-time setup required (run once on server):
    pip install "scrapling[all]"
    scrapling install

This downloads browser binaries. On Render, add this as a build command
step AFTER pip install in your render.yaml or build script.
"""

import asyncio
from typing import Optional
from loguru import logger

try:
    from scrapling.fetchers import StealthyFetcher, AsyncFetcher
    SCRAPLING_AVAILABLE = True
except ImportError:
    SCRAPLING_AVAILABLE = False
    logger.warning(
        "scrapling is not installed — Scrapling fetcher disabled. "
        "Run: pip install 'scrapling[all]' && scrapling install"
    )

ANTI_BOT_DOMAINS = {
    "skynews.com",
    "sky.com",
    "dailymaverick.co.za",
    "japantimes.co.jp",
    "thestar.com",
    "haaretz.com",
    "washingtonpost.com",
    "independent.co.uk",
}


def _needs_stealth(url: str) -> bool:
    return any(domain in url for domain in ANTI_BOT_DOMAINS)


def _stealthy_fetch_blocking(url: str):
    """
    Synchronous StealthyFetcher call.
    MUST only be called via asyncio.to_thread — never directly in async code.
    """
    return StealthyFetcher.fetch(
        url,
        headless=True,
        network_idle=True,
        disable_resources=True,
    )


async def fetch_with_scrapling(url: str) -> Optional[str]:
    """
    Async-safe Scrapling fetcher.
    Routes to StealthyFetcher for known anti-bot domains,
    AsyncFetcher for all others.
    Returns raw HTML string or None on failure.
    """
    if not SCRAPLING_AVAILABLE:
        return None

    try:
        if _needs_stealth(url):
            logger.debug(f"Scrapling: using StealthyFetcher for {url[:60]}")
            page = await asyncio.to_thread(_stealthy_fetch_blocking, url)
        else:
            logger.debug(f"Scrapling: using AsyncFetcher for {url[:60]}")
            page = await AsyncFetcher.get(url)

        if page.status == 200:
            # scrapling 0.2.x uses .html_content (TextHandler), not .html
            html_str = str(page.html_content)
            if len(html_str) > 500:
                logger.info(f"✅ Scrapling fetch success: {url[:60]}")
                return html_str

        logger.debug(
            f"Scrapling: status={page.status}, "
            f"length={len(str(page.html_content))} for {url[:60]}"
        )
    except Exception as e:
        logger.debug(f"Scrapling fetch error for {url}: {e}")

    return None
