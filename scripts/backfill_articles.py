#!/usr/bin/env python3
"""
Backfill article and prompt files from MongoDB.

Reads generated articles from MongoDB and writes them to:
  output/articles/YYYY-MM-DD/<session>_<idx>_<slug>.md
  output/prompts/YYYY-MM-DD/<session>_<idx>_<slug>_prompt.md

Safe to re-run — skips files that already exist.

Usage:
    python scripts/backfill_articles.py                  # backfill all generated articles
    python scripts/backfill_articles.py --date 2026-04-04  # specific date only
    python scripts/backfill_articles.py --session SCRAPE_20260404_082020_8142TP5C
"""

import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.database.mongo_client import MongoDBClient


def make_slug(topic: str) -> str:
    slug = topic.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    return slug[:50].rstrip('-')


def write_article_file(article: dict, articles_dir: Path, session_id: str, idx: int) -> Path:
    slug = make_slug(article.get('topic', 'untitled'))
    filename = f"{session_id}_{idx:02d}_{slug}.md"
    filepath = articles_dir / filename

    if filepath.exists():
        return filepath  # idempotent

    sources = article.get('sources_used', [])
    sources_str = ', '.join(sources) if sources else 'unknown'

    frontmatter = (
        f"---\n"
        f"session_id: {session_id}\n"
        f"topic: {article.get('topic', '')}\n"
        f"generated_at: {article.get('generated_at', '')}\n"
        f"word_count: {article.get('word_count', 0)}\n"
        f"source_count: {article.get('source_count', 0)}\n"
        f"source_quality: {article.get('source_quality', '')}\n"
        f"sources_used: [{sources_str}]\n"
        f"model_used: {article.get('model_used', '')}\n"
        f"---\n\n"
    )

    heading = article.get('heading', '')
    sub_heading = article.get('sub_heading', '')
    story = article.get('story', '')

    body = f"# {heading}\n\n"
    if sub_heading:
        body += f"### {sub_heading}\n\n"
    body += story

    filepath.write_text(frontmatter + body, encoding='utf-8')
    return filepath


def write_prompt_file(article: dict, prompts_dir: Path, session_id: str, idx: int):
    prompt_debug = article.get('prompt_debug')
    if not prompt_debug:
        return None

    slug = make_slug(article.get('topic', 'untitled'))
    filename = f"{session_id}_{idx:02d}_{slug}_prompt.md"
    filepath = prompts_dir / filename

    if filepath.exists():
        return filepath  # idempotent

    generated_at = article.get('generated_at', '')

    frontmatter = (
        f"---\n"
        f"session_id: {session_id}\n"
        f"topic: {article.get('topic', '')}\n"
        f"captured_at: {prompt_debug.get('captured_at', generated_at)}\n"
        f"model: {prompt_debug.get('model', '')}\n"
        f"source_count: {prompt_debug.get('source_count', 0)}\n"
        f"audit_quality: {prompt_debug.get('audit_quality', '')}\n"
        f"audit_sections: {prompt_debug.get('audit_sections', [])}\n"
        f"---\n\n"
    )

    system_msg = prompt_debug.get('system_message', '')
    user_prompt = prompt_debug.get('user_prompt', '')

    content = (
        frontmatter
        + "## SYSTEM MESSAGE\n\n"
        + system_msg
        + "\n\n---\n\n"
        + "## USER PROMPT\n\n"
        + user_prompt
        + "\n"
    )

    filepath.write_text(content, encoding='utf-8')
    return filepath


def backfill(date_filter: str = None, session_filter: str = None):
    db = MongoDBClient()
    if not db.connect():
        print("ERROR: Could not connect to MongoDB")
        sys.exit(1)

    query = {"pipeline_stage": "generated"}
    if date_filter:
        # Match generated_at field starting with the given date string
        query["generated_at"] = {"$regex": f"^{re.escape(date_filter)}"}
    if session_filter:
        query["session_id"] = session_filter

    projection = {"embedding": 0}
    articles = list(db.articles.find(query, projection).sort("generated_at", 1))

    if not articles:
        print(f"No generated articles found matching the query: {query}")
        return

    print(f"Found {len(articles)} generated article(s) to backfill.")

    articles_written = 0
    prompts_written = 0

    for article in articles:
        generated_at = article.get('generated_at', '')
        date_str = generated_at[:10] if generated_at else datetime.now().strftime('%Y-%m-%d')
        session_id = article.get('session_id', 'unknown')
        trend_idx = article.get('trend_index', 0)

        articles_dir = Path("output/articles") / date_str
        prompts_dir = Path("output/prompts") / date_str
        articles_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)

        art_path = write_article_file(article, articles_dir, session_id, trend_idx + 1)
        print(f"  article → {art_path}")
        articles_written += 1

        prompt_path = write_prompt_file(article, prompts_dir, session_id, trend_idx + 1)
        if prompt_path:
            print(f"  prompt  → {prompt_path}")
            prompts_written += 1
        else:
            print(f"  prompt  → (no prompt_debug data for this article)")

    print(f"\nDone. {articles_written} article file(s), {prompts_written} prompt file(s) written.")


def main():
    parser = argparse.ArgumentParser(description="Backfill article and prompt files from MongoDB")
    parser.add_argument('--date', help='Only backfill articles from this date (YYYY-MM-DD)')
    parser.add_argument('--session', help='Only backfill articles from this session ID')
    args = parser.parse_args()
    backfill(date_filter=args.date, session_filter=args.session)


if __name__ == '__main__':
    main()
