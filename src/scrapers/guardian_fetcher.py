# src/scrapers/guardian_fetcher.py

"""
Guardian Content API client for the OSI News Automation pipeline.

Fetches articles from The Guardian's official Content API and normalises
them to the pipeline's internal article shape.

Requires GUARDIAN_API_KEY environment variable.
Free key: https://open-platform.theguardian.com/
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import httpx
from loguru import logger

GUARDIAN_API_BASE = "https://content.guardianapis.com/search"

GUARDIAN_DEFAULT_PARAMS = {
    "show-fields": "bodyText,headline,byline,trailText",
    "order-by": "newest",
    "page-size": 10,
}


async def fetch_guardian_articles(
    query: str = "",
    section: str = "world",
    page_size: int = 10,
) -> List[Dict]:
    """
    Fetch articles from The Guardian's official Content API.
    Returns a list of article dicts normalised to the pipeline's
    internal article shape (heading, story, source_url, etc.).
    Returns an empty list if the API key is absent or the request fails.
    """
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        logger.warning(
            "GUARDIAN_API_KEY is not set — Guardian fetcher is disabled"
        )
        return []

    params = {
        **GUARDIAN_DEFAULT_PARAMS,
        "api-key": api_key,
        "section": section,
        "page-size": page_size,
    }
    if query:
        params["q"] = query

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0), follow_redirects=True
        ) as client:
            resp = await client.get(GUARDIAN_API_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("response", {}).get("results", [])
        articles = []
        for item in results:
            fields = item.get("fields", {})
            body = fields.get("bodyText", "").strip()
            if not body:
                continue  # skip articles with no extractable body

            byline = fields.get("byline", "")
            authors = [byline] if byline else []
            word_count = len(body.split())

            articles.append(
                {
                    "heading": item.get("webTitle", ""),
                    "story": body,
                    "source_url": item.get("webUrl", ""),
                    "source_name": "The Guardian",
                    "publish_date": item.get("webPublicationDate", ""),
                    "authors": authors,
                    "top_image": "",
                    "location": "",
                    "language": "en",
                    "scraped_at": datetime.utcnow().isoformat(),
                    "word_count": word_count,
                    "meta_keywords": [],
                    "meta_description": fields.get("trailText", ""),
                    "extraction_method": "guardian_api",
                }
            )
        logger.info(
            f"Guardian API returned {len(articles)} articles "
            f"[query='{query}', section='{section}']"
        )
        return articles

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Guardian API HTTP error {e.response.status_code}: {e}"
        )
    except Exception as e:
        logger.error(f"Guardian API unexpected error: {e}")

    return []
