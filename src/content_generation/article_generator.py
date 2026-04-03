"""
OSI News Automation System - Article Generator
===============================================
Generates comprehensive news articles from trend clusters using Groq API (LLaMA).
Synthesizes multiple source articles into one balanced, well-structured article.

Two-stage pipeline:
  Stage 1 — audit_source_material() audits what the sources can honestly support
  Stage 2 — generate_article() writes only what the audit approved
"""

import os
import sys
import time
import re
import instructor
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

from src.content_generation.models import AuditResult
from src.content_generation.prompt_builder import (
    extract_article_signals,
    detect_story_type_v2,
    build_synthesis_prompt_v2,
    validate_article_v2,
    parse_generated_article,
    build_dynamic_prompt,
    validate_article_dynamic,
    SYSTEM_MESSAGE_V3,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load environment variables
load_dotenv()


# ===========================================
# GROQ CLIENT INITIALIZATION
# ===========================================

_groq_client = None


def get_groq_client():
    """
    Get or initialize the Groq client.
    
    Returns:
        Groq client instance or None if API key missing.
    """
    global _groq_client
    
    if _groq_client is not None:
        return _groq_client
    
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        return None
    
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        logger.info("Groq client initialized successfully")
        return _groq_client
    except ImportError:
        logger.error("Groq package not installed. Run: pip install groq")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None


# ===========================================
# STAGE 1 — SOURCE MATERIAL AUDIT
# ===========================================

def audit_source_material(articles: List[Dict], topic: str, client) -> AuditResult:
    """
    Audit source material to determine what sections the sources can
    honestly support, before any article generation is attempted.

    Uses instructor + Groq to return a structured AuditResult.

    Args:
        articles: List of source article dicts.
        topic:    Trend topic string.
        client:   Groq client instance.

    Returns:
        AuditResult with available sections, quality assessment, and
        an honest word ceiling.
    """
    try:
        # Step 1 — Build compact source digest (up to 12 articles, 1500 chars each)
        # The larger sample gives the LLM enough material to accurately rate
        # source quality as "rich" for major stories with many contributors.
        digest_parts = []
        for n, article in enumerate(articles[:12], 1):
            source_name = article.get("source_name", "Unknown")
            heading = article.get("heading", "No headline")
            story = article.get("story", "")[:1500]
            digest_parts.append(f"Source {n} ({source_name}): {heading}\n{story}")
        source_digest = "\n\n".join(digest_parts)

        # Step 2 — Build the audit prompt
        audit_prompt = f"""Analyse these sources for the topic: {topic}

SOURCE MATERIAL:
{source_digest}

Return JSON with these exact keys:
- has_direct_quotes: true ONLY if quotation marks wrap ≥15 characters attributed to a named person
- has_named_sources: true if any person or institution is named anywhere in the sources
- has_statistics: true if any numeric figure is present
- has_future_event: true if any upcoming deadline, date, or decision is mentioned
- has_expert_opinion: true if any analytical or interpretive statement by a named expert exists
- has_impact_data: true if any consequence on another country, market, or population is explicitly stated
- primary_location: most specific city or country where events occur, or null if unclear
- source_quality: exactly one of:
    "rich"     → multiple sources, direct quotes, statistics, named people all present
    "adequate" → some facts and named sources present, limited or no direct quotes
    "thin"     → sparse facts, no direct quotes, no named people
- honest_word_ceiling: integer ceiling based on source quality:
    rich     → 700–900
    adequate → 400–600
    thin     → 180–350
- available_sections: always return this exact list regardless of source quality:
    ["lead_paragraph","key_facts","narrative","voices","analysis_context","implications","whats_next"]

RULES:
- Be ruthlessly honest about source quality — do not inflate ratings
- Return ONLY the raw JSON object, no preamble, no markdown fences"""

        # Step 3 — Call Groq using instructor
        model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        instructor_client = instructor.from_groq(client)
        audit_result = instructor_client.chat.completions.create(
            model=model,
            response_model=AuditResult,
            max_retries=2,
            temperature=0.1,
            max_tokens=600,  # Increased from 400 — handles larger 12-source digest
            messages=[
                {"role": "user", "content": audit_prompt}
            ],
        )

        # Step 4 — Log and return
        logger.info(
            f"📋 Audit: quality={audit_result.source_quality} | "
            f"sections={audit_result.available_sections} | "
            f"ceiling={audit_result.honest_word_ceiling}w"
        )
        return audit_result

    except Exception as e:
        # Step 5 — Fallback on any failure
        logger.warning(f"Audit call failed ({e}), using default audit result")
        default_result = AuditResult(
            available_sections=["what_happened", "key_facts", "background_context"],
            has_direct_quotes=False,
            has_named_sources=False,
            has_statistics=False,
            has_future_event=False,
            has_expert_opinion=False,
            has_impact_data=False,
            primary_location=None,
            source_quality="thin",
            honest_word_ceiling=400,
        )
        logger.info(
            f"📋 Audit: quality={default_result.source_quality} | "
            f"sections={default_result.available_sections} | "
            f"ceiling={default_result.honest_word_ceiling}w"
        )
        return default_result


# ===========================================
# DATELINE INFERENCE
# ===========================================

def infer_dateline(articles: List[Dict]) -> str:
    """
    Infer the dateline from source articles.
    
    Uses the most common location or defaults to NEW DELHI.
    
    Args:
        articles: List of source articles.
        
    Returns:
        Dateline string in uppercase.
    """
    locations = [a.get('location', 'Unknown') for a in articles]
    locations = [loc for loc in locations if loc and loc != 'Unknown']
    
    if not locations:
        return "NEW DELHI"
    
    location_counts = Counter(locations)
    most_common = location_counts.most_common(1)[0][0]
    
    return most_common.upper()


def format_timestamp(timezone: str = 'Asia/Kolkata') -> str:
    """
    Format current timestamp for article.
    
    Args:
        timezone: Timezone string.
        
    Returns:
        Formatted timestamp string.
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime('%A, %B %d, %Y, %I:%M %p IST')
    except ImportError:
        # Fallback without timezone
        return datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    except Exception:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ===========================================
# MAIN GENERATION FUNCTION
# ===========================================

def generate_article(
    trend: Dict, 
    target_words: int = 800,
    max_retries: int = 3,
    include_subheadings: bool = True
) -> Optional[Dict]:
    """
    Generate a comprehensive article from a trend cluster using Groq API.
    
    Uses a two-stage approach:
      Stage 1 — Audit source material to determine what can honestly be written
      Stage 2 — Write only what the audit approved, with dynamic section selection

    Args:
        trend: Trend dictionary with 'topic', 'articles', and 'keywords'.
        target_words: Target word count for generated article.
        max_retries: Maximum retry attempts on failure.
        include_subheadings: Whether to include subheadings.
        
    Returns:
        Generated article dictionary or None on failure.
    """
    # Step 1 — Extract source_articles and topic
    if not trend or 'articles' not in trend:
        logger.error("Invalid trend data provided")
        return None
    
    source_articles = trend.get('articles', [])
    topic = trend.get('topic', 'News Update')
    
    if not source_articles:
        logger.error("No source articles in trend")
        return None
    
    # Step 2 — Get Groq client
    client = get_groq_client()
    
    if not client:
        logger.warning("Groq client unavailable, using fallback")
        return generate_fallback_article(trend)
    
    # Step 3 — Audit source material
    audit = audit_source_material(source_articles, topic, client)
    
    # Step 4 — Hard stop: thin quality with insufficient sections
    if audit.source_quality == "thin" and len(audit.available_sections) < 2:
        logger.warning(
            f"⏭️ Skipping '{topic}' — audit found insufficient material "
            f"(quality=thin, sections={audit.available_sections})"
        )
        return None
    
    # Step 5 — Extract signals
    signals = extract_article_signals(source_articles)
    
    # Step 6 — Build dynamic prompt
    system_msg, user_prompt = build_dynamic_prompt(
        source_articles, topic, audit, signals
    )

    # Capture prompt for dashboard Prompts page
    prompt_debug = {
        "system_message": system_msg,
        "user_prompt":    user_prompt,
        "model":          os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
        "captured_at":    datetime.utcnow().isoformat(),
        "topic":          topic,
        "source_count":   len(source_articles),
        "audit_quality":  audit.source_quality,
        "audit_sections": audit.available_sections,
    }

    # Step 7 — Calculate max_tokens
    max_tokens = min(2300, int(audit.honest_word_ceiling * 1.5) + 200)
    
    logger.info(f"🖊️ Generating article for trend: '{topic}'")
    logger.info(f"   Sources: {len(source_articles)} articles")
    logger.info(f"   Audit: quality={audit.source_quality}, ceiling={audit.honest_word_ceiling}w")
    
    # Step 8 — Generation retry loop
    attempt = 0
    while attempt < max_retries:
        try:
            # a) Call Groq API
            model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.35,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            # b) Extract generated text
            generated_text = response.choices[0].message.content
            
            if not generated_text:
                logger.warning(f"Empty response from Groq (attempt {attempt + 1})")
                attempt += 1
                continue
            
            # c) Parse article
            article = parse_generated_article(generated_text)
            
            # d) Basic length check
            word_count = len(article.get("story", "").split())
            if word_count < 150:
                logger.warning(f"Article too short ({word_count} words), retrying...")
                attempt += 1
                continue
            
            # e) Validate with dynamic validator
            validation = validate_article_dynamic(article, audit)
            
            # f) Log all warnings
            for warning in validation.get("warnings", []):
                logger.warning(f"Article validation warning: {warning}")
            
            # g) Handle validation failures
            if not validation["passes"]:
                for failure in validation["failures"]:
                    logger.warning(f"Article validation failure: {failure}")
                if attempt < max_retries - 1:
                    attempt += 1
                    continue
                else:
                    logger.error(
                        "Article failed validation after all retries"
                    )
                    return None
            
            # h) Validation passed — add metadata
            article.update({
                "dateline": audit.primary_location.upper() if audit.primary_location else "INTERNATIONAL",
                "topic": topic,
                "timestamp": format_timestamp(),
                "sources_used": list({a.get("source_name", "Unknown") for a in source_articles}),
                "source_count": len(source_articles),
                "word_count": len(article.get("story", "").split()),
                "keywords": trend.get("keywords", [])[:10],
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": model,
                "source_quality": audit.source_quality,
                "audit_sections": audit.available_sections,
                "validation_warnings": validation.get("warnings", []),
                "prompt_debug": prompt_debug,
            })
            
            # i) Log success
            logger.info(f"✅ Generated article: '{article['heading'][:60]}...'")
            logger.info(
                f"   Words: {article['word_count']} | "
                f"Quality: {audit.source_quality} | "
                f"Sections: {audit.available_sections}"
            )
            
            # j) Return article
            return article
        
        except Exception as e:
            error_msg = str(e)
            
            # Rate limit: wait and retry WITHOUT consuming a retry slot
            if '429' in error_msg or 'rate' in error_msg.lower():
                logger.warning(
                    f"⏳ Rate limited by Groq — waiting 65s for quota reset "
                    f"(attempt {attempt + 1}/{max_retries} preserved, "
                    f"retry slot NOT consumed)..."
                )
                time.sleep(65)
                continue  # no attempt += 1, by design
            
            # Timeout: counts as an attempt
            if 'timeout' in error_msg.lower():
                logger.warning(f"Groq API timeout on attempt {attempt + 1}")
                attempt += 1
                continue
            
            # All other errors
            logger.error(f"Generation error (attempt {attempt + 1}): {e}")
            attempt += 1
            
            if attempt >= max_retries:
                logger.error("Max retries reached, using fallback")
                return generate_fallback_article(trend)
    
    # Step 9 — After loop exhausted
    return generate_fallback_article(trend)


def generate_fallback_article(trend: Dict) -> Optional[Dict]:
    """
    Generate a simple fallback article without LLM.
    
    Used when Groq API is unavailable or fails.
    
    Args:
        trend: Trend dictionary with articles.
        
    Returns:
        Simple article dictionary.
    """
    try:
        source_articles = trend.get('articles', [])
        topic = trend.get('topic', 'News Update')
        
        if not source_articles:
            return None
        
        # Create simple headline
        heading = f"Multiple Sources Report on {topic}"
        
        # Combine article summaries
        story_parts = [f"**{topic}**\n"]
        
        dateline = infer_dateline(source_articles)
        story_parts.append(f"{dateline}, {datetime.now().strftime('%B %d')} – ")
        story_parts.append(f"Multiple news sources are reporting on developments related to {topic}.\n\n")
        
        for i, article in enumerate(source_articles[:5], 1):
            source = article.get('source_name', 'Unknown')
            headline = article.get('heading', '')
            preview = article.get('story', '')[:200]
            
            story_parts.append(f"## Report from {source}\n\n")
            story_parts.append(f"**{headline}**\n\n")
            story_parts.append(f"{preview}...\n\n")
        
        story = ''.join(story_parts)
        
        return {
            "heading": heading,
            "story": story,
            "dateline": dateline,
            "timestamp": format_timestamp(),
            "sources_used": [a.get('source_name', 'Unknown') for a in source_articles],
            "word_count": len(story.split()),
            "source_count": len(source_articles),
            "topic": topic,
            "keywords": trend.get('keywords', []),
            "generated_at": datetime.utcnow().isoformat(),
            "model_used": "fallback",
            "is_fallback": True
        }
        
    except Exception as e:
        logger.error(f"Fallback generation failed: {e}")
        return None


def generate_articles_for_trends(
    trends: List[Dict],
    target_words: int = 800,
    max_articles: int = 20
) -> List[Dict]:
    """
    Generate articles for multiple trends.
    
    Args:
        trends: List of trend dictionaries.
        target_words: Target word count per article.
        max_articles: Maximum number of articles to generate.
        
    Returns:
        List of generated article dictionaries.
    """
    generated = []
    
    for i, trend in enumerate(trends[:max_articles]):
        logger.info(f"\nGenerating article {i + 1}/{min(len(trends), max_articles)}...")
        
        article = generate_article(trend, target_words)
        
        if article:
            generated.append(article)
        
        # Small delay between generations to avoid rate limits
        if i < len(trends) - 1:
            time.sleep(2)
    
    logger.info(f"Generated {len(generated)} articles from {len(trends)} trends")
    return generated


# ===========================================
# TESTING
# ===========================================

def test_article_generator():
    """Test article generation with sample trend."""
    print("\n" + "="*60)
    print("🧪 Article Generator Test")
    print("="*60)
    
    # Check Groq API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("\n⚠️ GROQ_API_KEY not set in environment")
        print("   Set it in .env file or environment variables")
        print("   Get free API key at: https://console.groq.com/")
        print("\n   Testing fallback generation instead...\n")
    
    # Create test trend
    test_trend = {
        "topic": "Global Climate Summit",
        "keywords": ["climate", "summit", "emissions", "agreement"],
        "articles": [
            {
                "heading": "World leaders reach historic climate agreement",
                "story": "In a landmark decision, world leaders at the Global Climate Summit have agreed to reduce carbon emissions by 50% by 2030. The agreement covers over 190 countries and includes financial commitments to support developing nations in their transition to clean energy. Environmental groups have cautiously welcomed the deal.",
                "source_name": "BBC News",
                "location": "Paris"
            },
            {
                "heading": "Climate summit produces breakthrough on emissions",
                "story": "After days of intense negotiations, delegates at the climate summit have reached a breakthrough agreement on emissions targets. The deal sets binding targets for major polluters and establishes a new fund for climate adaptation. Critics say the targets don't go far enough to limit warming to 1.5 degrees.",
                "source_name": "Reuters",
                "location": "Paris"
            },
            {
                "heading": "Environmental groups react to climate deal",
                "story": "Environmental organizations have given mixed reactions to the new climate agreement. While some praised the historic nature of the deal, others criticized the lack of enforcement mechanisms. Youth activists called for more ambitious action to address the climate crisis.",
                "source_name": "The Guardian",
                "location": "London"
            }
        ]
    }
    
    print(f"📰 Test trend: {test_trend['topic']}")
    print(f"   Sources: {len(test_trend['articles'])} articles")
    print("-" * 40)
    
    article = generate_article(test_trend, target_words=600)
    
    if article:
        print(f"\n✅ Article generated successfully!")
        print(f"\n📝 Headline: {article['heading']}")
        print(f"📍 Dateline: {article['dateline']}")
        print(f"📊 Word count: {article['word_count']}")
        print(f"📰 Sources: {', '.join(article['sources_used'])}")
        print(f"🤖 Model: {article.get('model_used', 'unknown')}")
        
        # Show preview
        preview = article['story'][:500]
        print(f"\n📖 Preview:\n{preview}...")
    else:
        print("\n❌ Article generation failed")
    
    print("\n" + "="*60 + "\n")
    
    return article


if __name__ == "__main__":
    test_article_generator()
