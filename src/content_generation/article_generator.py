"""
OSI News Automation System - Article Generator
===============================================
Generates comprehensive news articles from trend clusters using Groq API (LLaMA).
Synthesizes multiple source articles into one balanced, well-structured article.
"""

import os
import sys
import time
import re
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

from loguru import logger
from dotenv import load_dotenv

from src.content_generation.prompt_builder import (
    extract_article_signals,
    detect_story_type_v2,
    build_synthesis_prompt_v2,
    validate_article_v2,
    parse_generated_article,
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
    
    Takes multiple source articles about a topic and synthesizes them
    into one well-structured, comprehensive news article using the V2
    signal-routed prompt pipeline.
    
    Args:
        trend: Trend dictionary with 'topic', 'articles', and 'keywords'.
        target_words: Minimum word count for generated article.
        max_retries: Maximum retry attempts on failure.
        include_subheadings: Whether to include subheadings.
        
    Returns:
        Generated article dictionary with:
        - heading: Article headline
        - story: Full article text
        - dateline: Location dateline
        - timestamp: Generation timestamp
        - sources_used: List of source names
        - word_count: Final word count
        - source_count: Number of source articles used
        - story_type: Detected story type name string
        - validation_warnings: List of warning strings (empty if none)
        
    Example:
        >>> article = generate_article(trend_data, target_words=800)
        >>> print(article['heading'])
        >>> print(f"Word count: {article['word_count']}")
    """
    if not trend or 'articles' not in trend:
        logger.error("Invalid trend data provided")
        return None
    
    source_articles = trend.get('articles', [])
    topic = trend.get('topic', 'News Update')
    
    if not source_articles:
        logger.error("No source articles in trend")
        return None
    
    # Get Groq client
    client = get_groq_client()
    
    if not client:
        logger.warning("Groq client unavailable, using fallback")
        return generate_fallback_article(trend)
    
    # Pre-processing: extract signals and detect story type (deterministic —
    # run once before retry loop, no value in repeating on retry)
    signals = extract_article_signals(source_articles)
    story_type_config = detect_story_type_v2(source_articles, topic, signals)
    
    # Dynamic max_tokens based on target word count
    max_tokens = min(2300, int(target_words * 1.45) + 300)
    
    logger.info(f"🖊️ Generating article for trend: '{topic}'")
    logger.info(f"   Sources: {len(source_articles)} articles")
    logger.info(f"   Story type: {story_type_config.get('name', 'general')}")
    
    # Attempt generation with retries
    # NOTE: 'attempt' is a manual counter (not a for-loop variable) so that
    # rate-limit waits do NOT consume a retry slot. Every code path that
    # represents a real failure increments attempt explicitly. Rate limit
    # hits sleep and loop back without incrementing.
    attempt = 0
    while attempt < max_retries:
        try:
            # Build prompt inside retry loop (dateline may change between
            # attempts if the clock rolls over, and this keeps it clean)
            system_msg, user_prompt, dateline, _story_type = build_synthesis_prompt_v2(
                articles=source_articles,
                topic=topic,
                signals=signals,
                story_type_config=story_type_config,
                target_words=target_words,
                include_facts_snapshot=include_subheadings,
            )

            # Call Groq API
            model = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
            # ── Prompt debug capture ──
            prompt_debug = {
                "model": model,
                "system_msg": system_msg,
                "user_prompt": user_prompt,
                "total_word_estimate": len(user_prompt.split()) + len(system_msg.split()),
                "story_type": story_type_config.get("name", "general"),
                "topic": topic,
                "source_count": len(source_articles),
                "attempt": attempt + 1,
                "captured_at": datetime.utcnow().isoformat(),
            }
            logger.info(
                f"Prompt built — model: {model} | "
                f"~{prompt_debug['total_word_estimate']} words | "
                f"story type: {prompt_debug['story_type']}"
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.28,
                max_tokens=max_tokens,
                top_p=0.9,
            )

            # Extract generated content
            generated_text = response.choices[0].message.content

            if not generated_text:
                logger.warning(f"Empty response from Groq (attempt {attempt + 1})")
                attempt += 1
                continue

            # Parse article
            article = parse_generated_article(generated_text)
            article["prompt_debug"] = prompt_debug

            # Basic length check before full validation
            word_count = len(article['story'].split())

            if word_count < 300:
                logger.warning(f"Article too short ({word_count} words), retrying...")
                attempt += 1
                continue

            # V2 post-generation validation
            validation = validate_article_v2(article, topic, signals, story_type_config)

            if not validation["passes"]:
                for failure in validation["failures"]:
                    logger.warning(f"Article validation failure: {failure}")
                if attempt < max_retries - 1:
                    attempt += 1
                    continue
                else:
                    logger.error(
                        "Article failed validation after all retries — returning None"
                    )
                    return None

            # Log warnings (do not block upload)
            for warning in validation.get("warnings", []):
                logger.warning(f"Article validation warning: {warning}")

            # Add metadata
            article.update({
                "dateline":             dateline,
                "topic":                topic,
                "timestamp":            format_timestamp(),
                "sources_used":         list({a.get("source_name", "Unknown") for a in source_articles}),
                "source_count":         len(source_articles),
                "word_count":           len(article["story"].split()),
                "keywords":             trend.get("keywords", [])[:10],
                "generated_at":         datetime.utcnow().isoformat(),
                "model_used":           model,
                "story_type":           story_type_config.get("name", "general"),
                "validation_warnings":  validation.get("warnings", []),
            })

            logger.info(f"✅ Generated article: '{article['heading'][:50]}...'")
            logger.info(f"   Word count: {article['word_count']}")
            logger.info(f"   Dateline: {article['dateline']}")

            return article

        except Exception as e:
            error_msg = str(e)

            # ── Rate limit: wait and retry WITHOUT consuming a retry slot ──
            # Groq returns 429 when the per-minute token quota is exceeded.
            # Sleep 65s to let the 60s window reset, then retry the same
            # attempt index. attempt is NOT incremented here — this is
            # intentional and the core of this fix.
            if '429' in error_msg or 'rate' in error_msg.lower():
                logger.warning(
                    f"⏳ Rate limited by Groq — waiting 65s for quota reset "
                    f"(attempt {attempt + 1}/{max_retries} preserved, "
                    f"retry slot NOT consumed)..."
                )
                time.sleep(65)
                continue  # ← no attempt += 1 here, by design

            # ── Groq timeout: counts as an attempt ──
            if 'timeout' in error_msg.lower():
                logger.warning(f"Groq API timeout on attempt {attempt + 1}")
                attempt += 1
                continue

            # ── All other errors: log and consume attempt ──
            logger.error(f"Generation error (attempt {attempt + 1}): {e}")
            attempt += 1

            if attempt >= max_retries:
                logger.error("Max retries reached, using fallback")
                return generate_fallback_article(trend)

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
    max_articles: int = 5
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
