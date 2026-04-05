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
# GROQ CLIENT POOL — MULTI-KEY ROTATION
# ===========================================
# Keys are loaded from GROQ_API_KEY (primary), GROQ_API_KEY_2, _3, _4 …
# The system rotates to the next key automatically when:
#   • Transient rate limit persists for 270 s (3 × 90 s waits)
#   • Daily token quota (TPD) is exhausted — immediate rotation, no wait

_groq_clients: List = []        # ordered pool of Groq client instances
_current_key_index: int = 0     # index of the currently active client


def _build_groq_clients() -> List:
    """
    Build the Groq client pool from all GROQ_API_KEY* environment variables.
    Reads GROQ_API_KEY (key 1), then GROQ_API_KEY_2, GROQ_API_KEY_3, …
    """
    from groq import Groq

    keys = []
    primary = os.getenv('GROQ_API_KEY')
    if primary:
        keys.append(primary)
    for i in range(2, 20):
        k = os.getenv(f'GROQ_API_KEY_{i}')
        if k:
            keys.append(k)

    if not keys:
        logger.error("No GROQ_API_KEY* variables found in environment")
        return []

    clients = []
    for idx, key in enumerate(keys):
        try:
            clients.append(Groq(api_key=key))
            logger.info(f"Groq client {idx + 1}/{len(keys)} ready (…{key[-6:]})")
        except Exception as e:
            logger.warning(f"Could not init Groq client {idx + 1} (…{key[-6:]}): {e}")

    logger.info(f"Groq key pool: {len(clients)} key(s) available")
    return clients


def get_groq_client():
    """Return the currently active Groq client, building the pool on first call."""
    global _groq_clients, _current_key_index

    if not _groq_clients:
        try:
            _groq_clients = _build_groq_clients()
        except ImportError:
            logger.error("Groq package not installed. Run: pip install groq")
            return None

    if not _groq_clients:
        return None

    return _groq_clients[_current_key_index % len(_groq_clients)]


def rotate_groq_key(reason: str = "") -> bool:
    """
    Rotate to the next Groq API key in the pool.

    Returns:
        True  — a different key is now active.
        False — only one key in pool; rotation not possible.
    """
    global _groq_clients, _current_key_index

    if not _groq_clients or len(_groq_clients) < 2:
        logger.warning("Groq key rotation requested but pool has only one key.")
        return False

    old_idx = _current_key_index
    _current_key_index = (_current_key_index + 1) % len(_groq_clients)
    logger.warning(
        f"🔄 Groq key rotated: [{old_idx + 1} → {_current_key_index + 1}] "
        f"of {len(_groq_clients)} available. Reason: {reason}"
    )
    return True


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
            # Aligned with extract_article_signals() which also reads 2000 chars.
            # Mismatched lengths caused has_direct_quotes to be False for quotes
            # appearing between chars 1500-2000, contradicting signal extraction.
            story = article.get("story", "")[:2000]
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

        # Step 3 — Call Groq using JSON mode (avoids tool_use_failed from function-calling)
        model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        import json as _json
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
            messages=[
                {"role": "user", "content": audit_prompt}
            ],
        )
        data = _json.loads(response.choices[0].message.content)
        audit_result = AuditResult(**data)

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
# RETRY REPAIR CLASSIFIER
# ===========================================

def _build_repair_instructions(failures: list, word_ceiling: int) -> str:
    """
    Map validation failure messages to targeted repair instructions
    for the retry prompt. Each failure pattern gets a specific,
    actionable instruction — not a generic "try again".
    """
    instructions = []

    for failure in failures:
        f_lower = failure.lower()

        if "ceiling" in f_lower:
            instructions.append(
                f"• WORD CEILING: Your previous response exceeded "
                f"{word_ceiling} words. For this attempt, write "
                f"fewer paragraphs — cut the one that adds the least "
                f"new information. Stop writing the moment you reach "
                f"{word_ceiling} words. Do not summarise at the end."
            )

        elif "banned" in f_lower or any(
            phrase in f_lower for phrase in [
                "historic", "unprecedented", "crucial",
                "landmark", "sparking", "amid"
            ]
        ):
            instructions.append(
                "• BANNED PHRASE: Your previous response contained "
                "a banned phrase. Scan every sentence before writing "
                "it. If you are about to write 'historic', "
                "'unprecedented', 'crucial', 'landmark', 'sparking', "
                "or 'amid' — stop and replace it with a specific "
                "verified fact from the source material instead."
            )

        elif "short" in f_lower or "150" in f_lower:
            instructions.append(
                "• MINIMUM LENGTH: Your previous response was too short. "
                "Develop the development and context sections further "
                "using confirmed facts from SOURCE MATERIAL. "
                "Do not add speculation — add confirmed detail."
            )

        elif "headline" in f_lower:
            instructions.append(
                "• HEADLINE: Your previous response had a missing or "
                "too-short headline. Write a headline of at least "
                "6 words in active voice: who did what."
            )

        elif "forbidden generic section header" in f_lower:
            instructions.append(
                "• SECTION HEADER: Your previous response used a generic "
                "section header that is forbidden. Replace it with a header "
                "that names the specific angle of THIS story — for example, "
                "not '## The Human Cost' but '## How a Drone Hit a Market "
                "at 9:50 a.m. in Nikopol'. The header must be unique to "
                "this article and could not appear in a different story."
            )

        else:
            # Catch-all for any future failure type
            instructions.append(
                f"• GENERAL FAILURE: {failure[:120]}"
            )

    if not instructions:
        return "Review all rules in the system message and retry."

    return "\n".join(instructions)


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
    # 1.35 tokens/word is empirical for Llama 3.3 on news prose.
    # HEADLINE_OVERHEAD covers headline + subheadline + dateline tokens.
    # This makes max_tokens a hard enforcement of the audit ceiling,
    # not just a generous budget.
    TOKENS_PER_WORD = 1.35
    HEADLINE_OVERHEAD = 80
    max_tokens = int(audit.honest_word_ceiling * TOKENS_PER_WORD) + HEADLINE_OVERHEAD
    logger.debug(f"max_tokens set to {max_tokens} for ceiling {audit.honest_word_ceiling}w ({audit.source_quality})")
    
    logger.info(f"🖊️ Generating article for trend: '{topic}'")
    logger.info(f"   Sources: {len(source_articles)} articles")
    logger.info(f"   Audit: quality={audit.source_quality}, ceiling={audit.honest_word_ceiling}w")
    
    # Step 8 — Generation retry loop
    # keys_tried_this_article tracks which key indices have been exhausted so
    # we avoid cycling back to an already-burned key within the same article.
    attempt = 0
    keys_tried_this_article: set = set()
    last_validation_failures: list = []
    # Tracks failure reasons from the previous attempt so the
    # retry can send a targeted repair instruction.
    while attempt < max_retries:
        try:
            # a) Compute per-attempt token budget and repair preamble.
            # 10% reduction per retry adds hardware-level pressure on top of
            # the textual repair instruction. Floor of 200 prevents the model
            # being cut off so hard it cannot produce coherent output.
            attempt_max_tokens = max(
                200,
                int(max_tokens * (0.9 ** attempt))
            )

            if attempt == 0 or not last_validation_failures:
                current_user_prompt = user_prompt
            else:
                repair_instructions = _build_repair_instructions(
                    last_validation_failures,
                    audit.honest_word_ceiling
                )
                retry_preamble = (
                    f"⚠️ PREVIOUS ATTEMPT FAILED — DO NOT REPEAT THESE ERRORS:\n"
                    f"{repair_instructions}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                )
                current_user_prompt = retry_preamble + user_prompt
                logger.warning(
                    f"🔁 Retry {attempt}/{max_retries - 1} — "
                    f"injecting repair instructions: "
                    f"{last_validation_failures[:2]}"
                )

            # b) Call Groq API — always fetch current active client so rotation
            #    picked up inside the exception handler is immediately visible.
            client = get_groq_client()
            if not client:
                logger.error("No Groq client available, aborting generation")
                return generate_fallback_article(trend)
            model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": current_user_prompt},
                ],
                temperature=0.35,
                max_tokens=attempt_max_tokens,
                top_p=0.95,
            )
            
            # c) Extract generated text
            generated_text = response.choices[0].message.content

            if not generated_text:
                logger.warning(f"Empty response from Groq (attempt {attempt + 1})")
                attempt += 1
                continue

            # c2) Post-generation phrase substitutions — applied before validation
            # Replaces the most persistent banned phrases with safe semantic equivalents.
            # Only exact case-preserved matches; does not touch "historical", "Unprecedented" etc.
            _PHRASE_SUBS = [
                (" sparking ", " prompting "),
                (" sparking\n", " prompting\n"),
                ("sparking ", "prompting "),
                (" amid ", " as "),
                (" historic ", " notable "),
                (" historic\n", " notable\n"),
                ("historic ", "notable "),
                (" unprecedented ", " unparalleled "),
                ("unprecedented ", "unparalleled "),
                (" crucial ", " essential "),
                ("crucial ", "essential "),
                ("highlighted the need", "pointed to the need"),
                ("raises questions about", "draws attention to"),
            ]
            for _old, _new in _PHRASE_SUBS:
                generated_text = generated_text.replace(_old, _new)

            # c3) Fix subtitle heading level — LLM sometimes writes `# Subtitle` or
            # `## Subtitle` inside a section body instead of the required `### Subtitle`.
            # Correct any single-hash line that appears after the article headline.
            _fixed_lines = []
            _headline_seen = False
            for _line in generated_text.splitlines():
                if not _headline_seen and _line.startswith("# ") and not _line.startswith("## "):
                    _headline_seen = True
                    _fixed_lines.append(_line)
                elif _headline_seen and _line.startswith("# ") and not _line.startswith("## "):
                    # Single-hash subtitle inside body — promote to ###
                    _fixed_lines.append("##" + _line)
                else:
                    _fixed_lines.append(_line)
            generated_text = "\n".join(_fixed_lines)

            # c4) Strip forbidden generic section headers — LLM may output ## Lead etc.
            # despite FORMAT RULE. Custom story-specific ## headers are now intentional
            # and must NOT be stripped; only these specific blueprint labels are removed.
            _SECTION_HEADERS_TO_STRIP = {
                "## Lead", "## Background", "## What Happened", "## Voices",
                "## Analysis & Context", "## Implications", "## What's Next",
                "## Key Facts",
            }
            _cleaned_lines = [
                _l for _l in generated_text.splitlines()
                if _l.strip() not in _SECTION_HEADERS_TO_STRIP
            ]
            generated_text = "\n".join(_cleaned_lines)

            # c5) Strip incomplete final sentence — ensures article ends on a clean
            # sentence boundary even when the LLM is cut by the token limit.
            _trimmed = generated_text.rstrip()
            if _trimmed and _trimmed[-1] not in {'.', '!', '?', '"', '\u201d', ')'}:
                _last_end = max(
                    _trimmed.rfind('.'),
                    _trimmed.rfind('!'),
                    _trimmed.rfind('?'),
                    _trimmed.rfind('\u201d'),
                )
                # Only trim if the cut point is in the last 30% of the text —
                # prevents accidentally gutting a genuinely short article.
                if _last_end > len(_trimmed) * 0.70:
                    generated_text = _trimmed[:_last_end + 1] + "\n"
                    logger.debug("b5) Trimmed incomplete final sentence from article.")

            # d) Parse article
            article = parse_generated_article(generated_text)

            # e) Basic length check
            word_count = len(article.get("story", "").split())
            if word_count < 150:
                logger.warning(f"Article too short ({word_count} words), retrying...")
                attempt += 1
                continue

            # f) Validate with dynamic validator
            validation = validate_article_dynamic(article, audit)

            # g) Log all warnings
            for warning in validation.get("warnings", []):
                logger.warning(f"Article validation warning: {warning}")

            # h) Handle validation failures
            if not validation["passes"]:
                for failure in validation["failures"]:
                    logger.warning(f"Article validation failure: {failure}")
                last_validation_failures = validation.get("failures", [])
                if attempt < max_retries - 1:
                    attempt += 1
                    continue
                else:
                    logger.error(
                        f"Article failed all {max_retries} validation attempts. "
                        f"Final failures: {last_validation_failures}"
                    )
                    return None
            
            # i) Validation passed — add metadata
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
                "honest_word_ceiling": audit.honest_word_ceiling,  # Stored so the dashboard can display ceiling vs actual word count.
                "audit_sections": audit.available_sections,
                "validation_warnings": validation.get("warnings", []),
                "prompt_debug": prompt_debug,
            })
            
            # j) Log success
            logger.info(f"✅ Generated article: '{article['heading'][:60]}...'")
            logger.info(
                f"   Words: {article['word_count']} | "
                f"Quality: {audit.source_quality} | "
                f"Sections: {audit.available_sections}"
            )
            
            # k) Return article
            return article
        
        except Exception as e:
            error_msg = str(e)

            # ── Rate-limit handling ──────────────────────────────────────────
            if '429' in error_msg or 'rate_limit_exceeded' in error_msg.lower() \
                    or ('rate' in error_msg.lower() and 'limit' in error_msg.lower()):

                # Detect daily TPD exhaustion:
                #   "tokens per day" in the message, or wait time is in hours
                #   e.g. "Please try again in 1h9m27.072s"
                is_daily_limit = (
                    'tokens per day' in error_msg.lower()
                    or 'tpd' in error_msg.lower()
                    or bool(re.search(r'try again in \d+h', error_msg.lower()))
                )

                if is_daily_limit:
                    # Daily quota gone — no point waiting; rotate key immediately.
                    logger.warning(
                        f"🚫 Daily TPD quota exhausted on Groq key "
                        f"[{_current_key_index + 1}/{len(_groq_clients or [None])}]. "
                        f"Rotating to next key immediately..."
                    )
                    keys_tried_this_article.add(_current_key_index)
                    all_keys_count = len(_groq_clients) if _groq_clients else 1
                    if len(keys_tried_this_article) >= all_keys_count:
                        logger.error("All Groq keys have hit their daily TPD limit.")
                        return generate_fallback_article(trend)
                    rotated = rotate_groq_key("daily TPD quota exhausted")
                    if not rotated:
                        logger.error("Key rotation failed (single key pool).")
                        return generate_fallback_article(trend)
                    attempt = 0   # fresh attempt counter for the new key
                    continue

                else:
                    # Transient rate limit — wait 90 s and consume an attempt.
                    # After max_retries × 90 s = 270 s, rotate key.
                    attempt += 1
                    logger.warning(
                        f"⏳ Rate limited by Groq (key {_current_key_index + 1}) — "
                        f"waiting 90s (attempt {attempt}/{max_retries}, "
                        f"270s total before key rotation)..."
                    )
                    time.sleep(90)
                    if attempt >= max_retries:
                        # 270 s elapsed — escalate to next key
                        keys_tried_this_article.add(_current_key_index)
                        all_keys_count = len(_groq_clients) if _groq_clients else 1
                        if len(keys_tried_this_article) >= all_keys_count:
                            logger.error("All Groq keys rate-limited after 270 s each.")
                            return generate_fallback_article(trend)
                        rotated = rotate_groq_key(
                            f"transient rate limit persisted for {max_retries * 90}s"
                        )
                        if rotated:
                            attempt = 0  # fresh attempt counter for the new key
                    continue

            # ── Timeout ─────────────────────────────────────────────────────
            if 'timeout' in error_msg.lower():
                logger.warning(
                    f"⏱ Groq API timeout on attempt {attempt + 1}/{max_retries}"
                )
                attempt += 1
                continue

            # ── All other errors ─────────────────────────────────────────────
            logger.error(f"Generation error (attempt {attempt + 1}): {e}")
            attempt += 1
            if attempt >= max_retries:
                logger.error("Max retries reached, using fallback")
                return generate_fallback_article(trend)

    # Step 9 — After loop exhausted without success
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
