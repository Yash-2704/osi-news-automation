# Integration Notes — Guardian API, Fundus, Scrapling

Three-module integration into the OSI News Automation async scraping pipeline.
Applied **6 change sets** across 5 new/modified files.

---

## Section 1: Integration Walkthrough

Trace a single article fetch for a **Reuters URL** through the complete
new fetch chain in execution order:

### Step 1: Entry point
`_fetch_and_extract(url, source_name="Reuters")` is called from
`_bounded_fetch()` inside `_scrape_news_batch_async()`. The `source_name`
is passed from `source.get('name', 'Unknown')` in the batch loop.

### Step 2: Tier 0 — Fundus publisher-specific parser
`fetch_with_fundus(url, "Reuters")` is called.
- The source name is normalised to lowercase: `"reuters"`
- Fundus's `_PUBLISHER_MAP` maps `"reuters"` → `PublisherCollection.us.Reuters`
- `Crawler(publisher)` crawls the URL synchronously inside `asyncio.to_thread()`
- If Fundus extracts a body with >150 chars, the function returns immediately
  with a fully-formed article dict (`extraction_method: "fundus"`). The
  remaining tiers are **skipped**.
- If Fundus fails (import guard, no body, exception), returns `None` → 
  execution continues to Tier 1.

### Step 3: Tier 1 — Scrapling anti-bot bypass
`fetch_with_scrapling(url)` is called.
- Reuters is **not** in `ANTI_BOT_DOMAINS`, so `AsyncFetcher.fetch(url)` 
  is used (not StealthyFetcher).
- If Scrapling returns HTML with `status==200` and `len(html)>500`, the
  raw HTML is assigned to `html_content` and passed to the existing
  trafilatura extraction path (Tier 2's extraction, not its fetch).
- If Scrapling is not installed or fails, `html_content` is set to `None`
  → execution continues to the existing fetch path.

### Step 4: Tier 2 — Existing curl_cffi / httpx fetch
Only reached if `html_content is None` (both Tier 0 and Tier 1 produced
no results).
- `_fetch_html(url)` is called with tenacity retry (3 attempts, exponential
  backoff). Uses curl_cffi with Chrome TLS impersonation if available,
  httpx as fallback.
- The resulting HTML passes through trafilatura extraction, metadata
  extraction, language detection, and location extraction — the original
  pipeline path, completely unchanged.

### Step 5: Guardian path
`fetch_guardian_articles()` is a **parallel source ingestion path**, not
part of the per-URL fetch chain described above. It fetches articles
directly from The Guardian's Content API and returns pre-formed article
dicts. It should be called separately (e.g., in the orchestration layer)
and its results merged into the article list alongside scraped articles.
It is **not** called from `_fetch_and_extract()`.

---

## Section 2: Verified Bugs — All Fixed

```
BUG-1 ✅ FIXED
File: src/scrapers/fundus_fetcher.py + batch_scraper.py
Description: Fundus heading was taken from body's first line, not actual title.
Fix Applied: _crawl_blocking() now returns (title, body) tuple. Tier 0 in
  batch_scraper.py unpacks the tuple and uses article.title when available.
```

```
BUG-2 ✅ FIXED
File: src/scrapers/guardian_fetcher.py
Description: Return dict keys didn't match pipeline schema (title/body/url 
  vs heading/story/source_url).
Fix Applied: Changed all return keys to match pipeline schema: heading,
  story, source_url, publish_date, authors (list), plus added word_count,
  scraped_at, extraction_method, location, meta_description, top_image.
```

```
BUG-3 ✅ FIXED
File: src/scrapers/fundus_fetcher.py
Description: article.body.text would AttributeError since body is a string.
Fix Applied: Changed to `article.body if article.body else ""`.
```

```
BUG-4 ✅ FIXED
File: src/scrapers/scrapling_fetcher.py
Description: page.html might return an Adaptor object, not a raw string.
Fix Applied: Added `str(page.html) if not isinstance(page.html, str) else page.html`
  to ensure a string is always returned to trafilatura.
```

```
BUG-5 ✅ FIXED
File: src/scrapers/batch_scraper.py
Description: Tier 0 Fundus path hardcoded language="en".
Fix Applied: Now runs langdetect on fundus_body, consistent with Tier 2.
```

```
BUG-6 ⚠️ DEPLOYMENT RISK (not a code bug)
File: render.yaml
Description: `scrapling install` downloads browser binaries which may
  exceed Render free-tier disk limits.
Mitigation: Test the build on Render. If it exceeds limits, remove 
  `scrapling install` from buildCommand — StealthyFetcher will be 
  disabled at runtime with a logged warning; AsyncFetcher still works.
```

---

## Section 3: Environment Variables Checklist

| Variable | Required | Default | Where to set |
|---|---|---|---|
| `GUARDIAN_API_KEY` | Yes (for Guardian) | `None` — disables Guardian fetcher | Render dashboard + `.env` |
| `GROQ_API_KEY` | Yes (for LLM) | None | Render dashboard + `.env` |
| `MONGO_URI` | Yes (for DB) | None | Render dashboard + `.env` |
| `MONGO_DB_NAME` | Yes (for DB) | `osi_news_automation` | Render dashboard + `.env` |
| `HF_ACCESS_TOKEN` | Yes (for images) | None | Render dashboard + `.env` |
| `CLOUDINARY_CLOUD_NAME` | Yes (for image hosting) | None | Render dashboard |
| `CLOUDINARY_API_KEY` | Yes (for image hosting) | None | Render dashboard |
| `CLOUDINARY_API_SECRET` | Yes (for image hosting) | None | Render dashboard |
| `USER_AGENT` | No | `RobinOSI-Bot/1.0` | `.env` |
| `REQUEST_TIMEOUT_SECONDS` | No | `30` | `.env` |
| `MIN_ARTICLE_WORDS` | No | `50` | `.env` |
| `MAX_ARTICLES_PER_RUN` | No | `5` | Render dashboard |

**Interaction notes:**
- `GUARDIAN_API_KEY` is completely independent of existing variables. If absent,
  the Guardian fetcher logs a warning and returns an empty list — no other
  functionality is affected.
- `fundus` and `scrapling` do not require any environment variables. They are
  library-level dependencies that work out of the box once installed.

---

## Section 4: Testing Checklist

### Guardian API test
- [ ] Set `GUARDIAN_API_KEY` in `.env` with a valid free key
- [ ] Run a quick test:
  ```python
  import asyncio
  from src.scrapers.guardian_fetcher import fetch_guardian_articles
  articles = asyncio.run(fetch_guardian_articles(section="world", page_size=3))
  print(f"Got {len(articles)} articles")
  for a in articles:
      print(f"  - {a['title'][:60]}")
  ```
- [ ] Verify: returns a non-empty list, each dict has keys `title`, `body`, `url`, `source_name`, `published_at`, `author`

### Fundus test
- [ ] Test with **Reuters** first — it's a Tier 1 wire service already in the pipeline, high volume, and one of the most reliable Fundus publishers
- [ ] Run:
  ```python
  import asyncio
  from src.scrapers.fundus_fetcher import fetch_with_fundus
  body = asyncio.run(fetch_with_fundus("https://www.reuters.com/world/", "Reuters"))
  print(f"Body length: {len(body) if body else 0}")
  ```
- [ ] Verify: returns a non-empty string or `None` (graceful failure)

### Scrapling test
- [ ] Test with `independent.co.uk` first — it's in `ANTI_BOT_DOMAINS` and will exercise the `StealthyFetcher` path
- [ ] Run:
  ```python
  import asyncio
  from src.scrapers.scrapling_fetcher import fetch_with_scrapling
  html = asyncio.run(fetch_with_scrapling("https://www.independent.co.uk/news"))
  print(f"HTML length: {len(html) if html else 0}")
  ```
- [ ] Verify: returns HTML string with length > 500, or `None` with a debug log

### Regression test
- [ ] Run the existing batch scraper with a previously-working source (e.g., AP News via RSS):
  ```python
  from src.scrapers.batch_scraper import scrape_news_batch
  articles = scrape_news_batch(max_articles=3, prefer_rss=True)
  for a in articles:
      print(f"{a['source_name']}: {a['heading'][:50]}")
  ```
- [ ] Verify: articles are returned with the same schema as before the integration
- [ ] Verify: no `ImportError` or `KeyError` in logs even if `fundus`/`scrapling` are not installed

### Full integration test
- [ ] Run a complete pipeline cycle: `python run_automation.py --mode once`
- [ ] Check logs for:
  - `✅ Fundus extraction success` lines (Tier 0 hits)
  - `✅ Scrapling fetch success` lines (Tier 1 hits)
  - `✅ Scraped:` lines (Tier 2 existing path still working)
  - No unhandled exceptions from the new modules
