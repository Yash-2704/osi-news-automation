"""
OSI News Automation – Prompt Builder (V2)
==========================================
Builds editorially-structured prompts for Groq / LLaMA article synthesis
using a 10-section editorial format with signal-routed content.

Provides:
    extract_article_signals    – mines quotes, key facts, human angle from sources
    detect_story_type_v2       – classifies articles into one of 8 story types
    build_synthesis_prompt_v2  – returns (system_msg, user_prompt, dateline)
    validate_article_v2        – hard post-generation section/quality validator
    resolve_dateline           – LLM-first dateline with Counter fallback
    parse_generated_article    – splits LLM output into heading / sub_heading / story
    SYSTEM_MESSAGE_V3          – journalist persona (system role constant)
    STORY_TYPES_V2             – 8-type story taxonomy with per-type config
"""

import os
import re
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.content_generation.location_extractor import extract_location_and_category
from src.content_generation.models import AuditResult


# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL COMPILED REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Quote extraction: captures quoted text (20–200 chars) between curly or
# straight quote characters, optionally followed by an attribution verb
# and a capitalised speaker name.
#   Group 1 = quoted text
#   Group 2 = attribution verb (optional)
#   Group 3 = speaker name — one or more capitalised words (optional)
_QUOTE_RE = re.compile(
    r'[\u201c\u201d""]'           # opening quote (curly or straight)
    r'([^"\u201c\u201d\u201e]{20,200})'  # captured quote body: 20-200 chars
    r'[\u201c\u201d""]'           # closing quote
    r'(?:\s*,?\s*'                # optional separator
    r'(?:said|stated|told|confirmed|warned|added|noted|declared))?'  # attribution verb (optional non-capturing)
    r'(?:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))?'  # speaker name: capitalised words (optional)
)

# Key fact patterns — three groups:
# (a) Numeric claims with units (e.g. "500 people", "3.5 percent")
_KEYFACT_NUMERIC_RE = re.compile(
    r'\b(\d[\d,\.]*\s*(?:people|percent|billion|million|km|kilometers|'
    r'kilometres|years|months|troops|soldiers|casualties|deaths|injured|'
    r'wounded|displaced|refugees|tons|tonnes|dollars|euros|pounds))\b',
    re.IGNORECASE,
)

# (b) Full dates with month name + day + year (e.g. "March 21, 2024")
_KEYFACT_DATE_RE = re.compile(
    r'\b((?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2},?\s+\d{4})\b'
)

# (c) Named institutions ending in key suffixes
_KEYFACT_INSTITUTION_RE = re.compile(
    r'\b([A-Z][A-Za-z\s]{3,40}\s+'
    r'(?:Ministry|Government|Agency|Organisation|Organization|Authority|'
    r'Commission|Council|Department|Bureau|Committee))\b'
)

# Sentence boundary splitter — handles .!? followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Human-angle scoring keywords
_HUMAN_ANGLE_KEYWORDS = frozenset([
    'family', 'families', 'children', 'child', 'civilian', 'civilians',
    'resident', 'residents', 'community', 'communities', 'victim', 'victims',
    'survivor', 'survivors', 'refugee', 'refugees', 'worker', 'workers',
    'displaced', 'shelter', 'orphan', 'elderly', 'women', 'infant',
])

# Outlet name leak detection
_OUTLET_LEAK_RE = re.compile(
    r'\b(?:BBC|Reuters|CNN|Al\s*Jazeera|Associated\s*Press)\b',
    re.IGNORECASE,
)

# RSS scrapers frequently capture navigation text as speaker names
# via _QUOTE_RE's capitalised-word capture group. This blocklist
# maps known artifacts to "unnamed official" before quotes are
# stored or injected into generation prompts.
_INVALID_SPEAKERS: frozenset = frozenset({
    "stalled", "stalled negotiations", "related stories",
    "recommended stories", "read more", "developing story",
    "breaking news", "editor's note", "advertisement",
    "subscribe", "newsletter", "sign up", "follow us",
    "share", "comments", "loading", "updated", "published",
})

# Generic section headers that the model must never use. These labels
# name a category of information rather than the specific angle of the
# story being written. Defined here so validate_article_dynamic() can
# enforce them as hard failures and build_dynamic_prompt() can reference
# the same canonical list in future if needed.
_FORBIDDEN_HEADERS: frozenset = frozenset({
    "the escalating conflict", "the human cost",
    "the humanitarian crisis", "the response",
    "the diplomatic efforts", "the broader context",
    "the economic cost", "the international context",
    "the implications", "the potential consequences",
    "the road ahead", "what comes next", "looking ahead",
    "background", "what happened", "voices",
    "analysis & context", "key facts", "lead",
    "the scale of the attacks", "the response from",
    "the broader implications", "the international response",
    "the international community's response", "the economic cost of conflict",
    "the role of", "the future of", "the broader picture",
    "the context", "the significance", "the global implications",
})


# ═══════════════════════════════════════════════════════════════════
# STORY TYPE TAXONOMY — V2
# ═══════════════════════════════════════════════════════════════════

# DEPRECATED — replaced by two-stage audit approach. Do not call this function.
STORY_TYPES_V2: Dict = {
    "conflict": {
        "name": "conflict",
        "keywords": [
            "war", "military", "conflict", "attack", "airstrike", "bombing",
            "soldiers", "troops", "casualties", "ceasefire", "frontline",
            "weapons", "artillery", "invasion", "offensive", "defense",
            "militia", "insurgent", "battle",
        ],
        "sections": [
            "The Military Situation",
            "Civilian Impact",
            "International Response",
            "Strategic Analysis",
            "Prospects for Resolution",
        ],
        "quote_instruction": (
            "Prioritise voices from military officials, humanitarian agencies, "
            "and affected civilians."
        ),
        "impact_angle": (
            "Explore how the conflict reshapes regional alliances, refugee flows, "
            "and energy or trade corridors."
        ),
    },
    "humanitarian": {
        "name": "humanitarian",
        "keywords": [
            "humanitarian", "refugee", "displaced", "aid", "relief", "famine",
            "crisis", "shelter", "victims", "civilians", "suffering",
            "hunger", "malnutrition", "evacuation", "rescue", "donation",
            "volunteer", "camp", "migration",
        ],
        "sections": [
            "The Human Cost",
            "Aid and Relief Efforts",
            "Obstacles to Assistance",
            "Personal Testimonies",
            "Long-term Recovery",
        ],
        "quote_instruction": (
            "Prioritise voices from aid workers, affected families, and "
            "UN agency spokespersons."
        ),
        "impact_angle": (
            "Explore how the crisis strains neighbouring countries' resources "
            "and international aid budgets."
        ),
    },
    "political": {
        "name": "political",
        "keywords": [
            "election", "government", "president", "parliament", "minister",
            "political", "legislation", "vote", "policy", "opposition",
            "coalition", "reform", "diplomat", "sanctions", "summit",
            "treaty", "constitution", "campaign",
        ],
        "sections": [
            "The Development",
            "Political Landscape",
            "Stakeholder Positions",
            "Public Reaction",
            "What Comes Next",
        ],
        "quote_instruction": (
            "Prioritise voices from elected officials, party leaders, "
            "political analysts, and affected constituencies."
        ),
        "impact_angle": (
            "Explore how this political development shifts domestic power "
            "dynamics and international diplomatic relations."
        ),
    },
    "economic": {
        "name": "economic",
        "keywords": [
            "economy", "market", "gdp", "inflation", "stock", "trade",
            "financial", "investment", "currency", "recession", "growth",
            "employment", "industry", "revenue", "profit", "tariff",
            "interest rate", "budget", "fiscal",
        ],
        "sections": [
            "The Economic Event",
            "Market Response",
            "Transmission & Impact",
            "Policy & Intervention",
            "Historical Context",
        ],
        "quote_instruction": (
            "Prioritise voices from central bank officials, finance ministers, "
            "economists, and business leaders."
        ),
        "impact_angle": (
            "Explore how this economic event affects global supply chains, "
            "consumer prices, and investment confidence."
        ),
    },
    "scientific": {
        "name": "scientific",
        "keywords": [
            "study", "research", "findings", "scientist", "discovery",
            "breakthrough", "published", "journal", "peer-reviewed",
            "experiment", "data", "clinical", "medical", "vaccine",
            "laboratory", "hypothesis", "genome", "technology",
        ],
        "sections": [
            "The Discovery",
            "Scientific Context",
            "Methodology & Reliability",
            "Expert Reception",
            "Path Forward",
        ],
        "quote_instruction": (
            "Prioritise voices from lead researchers, peer reviewers, "
            "and independent subject-matter experts."
        ),
        "impact_angle": (
            "Explore how this discovery may change clinical practice, "
            "public health policy, or future research directions."
        ),
    },
    "social": {
        "name": "social",
        "keywords": [
            "trend", "social", "cultural", "generation", "adoption",
            "behavior", "demographic", "movement", "community", "society",
            "lifestyle", "millennials", "gen z", "viral", "protest",
            "rights", "equality", "activism",
        ],
        "sections": [
            "The Shift",
            "Who's Leading, Who's Resisting",
            "Institutional Response",
            "Speed & Scale",
            "Substance Assessment",
        ],
        "quote_instruction": (
            "Prioritise voices from community organisers, sociologists, "
            "and people directly affected by the shift."
        ),
        "impact_angle": (
            "Explore how this social change influences legislation, "
            "institutional norms, and neighbouring societies."
        ),
    },
    "disaster": {
        "name": "disaster",
        "keywords": [
            "earthquake", "flood", "hurricane", "tornado", "wildfire",
            "tsunami", "landslide", "cyclone", "typhoon", "eruption",
            "storm", "disaster", "emergency", "collapse", "destruction",
            "devastation", "rescue", "evacuation", "death toll",
        ],
        "sections": [
            "The Event",
            "Damage and Casualties",
            "Rescue and Response",
            "Infrastructure Impact",
            "Recovery Outlook",
        ],
        "quote_instruction": (
            "Prioritise voices from emergency services, disaster management "
            "agencies, and survivors."
        ),
        "impact_angle": (
            "Explore how the disaster affects regional infrastructure, "
            "insurance markets, and climate-resilience planning."
        ),
    },
    "general": {
        "name": "general",
        "keywords": [],  # fallback — no keywords to match
        "sections": [
            "What Happened",
            "Key Details",
            "Background & Context",
            "Reactions",
            "Looking Ahead",
        ],
        "quote_instruction": (
            "Include attributed statements from the most authoritative "
            "voices related to the story."
        ),
        "impact_angle": (
            "Explore broader implications for affected communities "
            "and relevant institutions."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════
# SYSTEM MESSAGE — V3
# ═══════════════════════════════════════════════════════════════════

# DEPRECATED — replaced by two-stage audit approach. Do not call this function.
SYSTEM_MESSAGE_V3: str = (
    "You are a senior international correspondent with twenty years of "
    "field reporting experience. You write for an educated general "
    "audience that expects accuracy, context, and prose that respects "
    "their intelligence.\n\n"

    "Before you write a single word of an article, you understand your "
    "material. You know what you have and what you do not have. You "
    "never fill gaps with memory or inference — you name the gap "
    "honestly and move on. A short truthful article is always more "
    "valuable than a long fabricated one.\n\n"

    "Your articles have shape. They begin with a hook that earns the "
    "reader's attention. They develop a central tension. They ground "
    "abstract events in real consequences. They close on the open "
    "question that remains — not a platitude, but the specific thing "
    "that will determine what happens next.\n\n"

    "You attribute statements to named people and institutions only "
    "when those names appear in your source material. You never infer "
    "a title or role from memory. If a source names a person without "
    "stating their role, you use their name only.\n\n"

    "You follow AP Style. You do not editorialize. When you offer "
    "analysis, you label it explicitly as analysis. You never use the "
    "words 'crucial', 'landmark', 'historic', or 'unprecedented' "
    "unless a source uses them and you are quoting directly."
)


# ═══════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_article_signals(articles: List[Dict]) -> Dict:
    """
    Mine structured signals from source articles for prompt routing.

    Extracts quotes, key facts, a human-angle sentence, and a
    de-duplicated source digest from up to 8 source articles.

    Args:
        articles: List of source article dicts, each with 'story',
                  'heading', 'source_name', 'location' keys.

    Returns:
        dict with keys:
            quotes       – list of {"text": str, "speaker": str} dicts (max 6)
            key_facts    – list of unique fact strings (max 10)
            human_angle  – str (longest qualifying sentence) or ""
            source_digest – str (formatted multi-source digest text)

    Example:
        >>> signals = extract_article_signals(source_articles)
        >>> print(signals["quotes"][0]["text"])
    """
    quotes: List[Dict[str, str]] = []
    key_facts: List[str] = []
    human_angle: str = ""
    human_angle_score: int = 0
    digest_parts: List[str] = []

    usable = articles[:8]

    for idx, article in enumerate(usable, 1):
        story = article.get("story", "")
        heading = article.get("heading", "No headline")
        source_name = article.get("source_name", "Unknown Source")
        location = article.get("location", "Unknown")

        # ── Extract quotes ──
        for match in _QUOTE_RE.finditer(story):
            quote_text = match.group(1).strip()
            # Fix 1 — Explainer-box filter: "What is X? X is ..." constructs
            # are editorial copy paste, not attributable quotes. Only reject
            # when >80 chars so short rhetorical questions ("What choice do we
            # have?") are preserved.
            if re.search(r'\?\s+[A-Z]', quote_text) and len(quote_text) > 80:
                continue
            open_pos = match.start()
            pre_context = story[max(0, open_pos - 80):open_pos]
            attribution_match = re.search(
                r'([A-Za-z][^.!?]{5,60}?'
                r'(?:said|stated|told|wrote|confirmed|warned|added'
                r'|noted|declared|announced|reported))\s*[,:]?\s*$',
                pre_context
            )
            if attribution_match:
                speaker = attribution_match.group(1).strip()
            elif match.group(2):
                speaker = match.group(2).strip()
            else:
                speaker = pre_context.strip()[-40:].strip(' ,"\'')

            # Drop quotes with no recoverable attribution — a quote
            # with no speaker is worse than no quote at all.
            if not speaker or len(speaker) < 4 or not any(c.isalpha() for c in speaker):
                continue

            # Sanitise speaker: reject RSS navigation artifacts
            if speaker.lower().strip() in _INVALID_SPEAKERS:
                continue

            # De-duplicate by checking if similar text already captured
            if not any(q["text"][:40] == quote_text[:40] for q in quotes):
                quotes.append({"text": quote_text, "speaker": speaker})

        # ── Extract key facts ──
        for pattern in (_KEYFACT_NUMERIC_RE, _KEYFACT_DATE_RE, _KEYFACT_INSTITUTION_RE):
            for match in pattern.finditer(story):
                fact = match.group(1).strip()
                if fact not in key_facts:
                    key_facts.append(fact)

        # ── Extract human angle ──
        sentences = _SENTENCE_SPLIT_RE.split(story)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in _HUMAN_ANGLE_KEYWORDS if kw in sentence_lower)
            # Keep the longest sentence that has the highest keyword score
            if score > human_angle_score or (
                score == human_angle_score and score > 0 and len(sentence) > len(human_angle)
            ):
                human_angle = sentence.strip()
                human_angle_score = score

        # ── Build source digest: 2000 chars per source, quotes kept in place ──
        # Quotes are not stripped from the digest — they remain in context so the
        # model sees them naturally within the source text. build_dynamic_prompt()
        # also surfaces them prominently via quote_block and raw_quotes_block,
        # injected BEFORE SOURCE MATERIAL so the model reads what it can quote
        # verbatim before it encounters the full raw context. Do not strip quotes
        # here without also removing those injection blocks in build_dynamic_prompt().
        snippet = story[:2000].strip()

        digest_parts.append(
            f"Source {idx} ({source_name}, {location}):\n"
            f"Headline: {heading}\n"
            f"Content: {snippet}"
        )

    # ── Cap and de-duplicate final lists ──
    quotes = quotes[:6]
    key_facts = key_facts[:10]

    source_digest = "\n\n".join(digest_parts) if digest_parts else "(no source material)"

    logger.debug(
        f"Signal extraction: {len(quotes)} quotes, {len(key_facts)} facts, "
        f"human_angle={'yes' if human_angle else 'no'}"
    )

    return {
        "quotes": quotes,
        "key_facts": key_facts,
        "human_angle": human_angle,
        "source_digest": source_digest,
    }


# ═══════════════════════════════════════════════════════════════════
# STORY TYPE DETECTION — V2
# ═══════════════════════════════════════════════════════════════════

def detect_story_type_v2(
    articles: List[Dict],
    topic: str,
    signals: Dict,
) -> Dict:
    # DEPRECATED — replaced by two-stage audit approach. Do not call this function.
    """
    Classify source articles into one of 8 story types using keyword
    scoring and signal-based boosts.

    Builds a combined text from the topic string and the first 400 chars
    of each article (max 6), scores keyword hits for each type, applies
    signal boosts, and returns the full config dict for the winning type.

    Args:
        articles: List of source article dicts.
        topic:    Trend topic string.
        signals:  Dict returned by extract_article_signals().

    Returns:
        Config dict from STORY_TYPES_V2 for the best-scoring type.
        Always includes 'name', 'keywords', 'sections',
        'quote_instruction', and 'impact_angle' keys.

    Example:
        >>> config = detect_story_type_v2(articles, "Iran conflict", signals)
        >>> print(config["name"])  # e.g. "conflict"
    """
    # Build combined text for keyword scanning
    # NOTE: topic label excluded — it is a thin auto-generated string that
    # dilutes keyword scores and causes topical clusters to fall below the
    # threshold of 3, defaulting to "general". Signal comes from headings
    # and story content only.
    combined = ""
    for article in articles[:6]:
        combined += article.get("heading", "").lower() + " "
        combined += article.get("story", "")[:400].lower() + " "

    # Score each type (skip 'general' — it has no keywords and is the fallback)
    scores: Dict[str, int] = {}
    for type_name, config in STORY_TYPES_V2.items():
        if type_name == "general":
            continue
        score = sum(1 for kw in config["keywords"] if kw in combined)
        scores[type_name] = score

    # --- Fix 2 — humanitarian and conflict boost guard ---
    # Signal boosts are gated: only applied when keyword score already
    # reaches threshold (>=3), preventing single-signal type overrides
    if signals.get("human_angle") and scores.get("humanitarian", 0) >= 3:
        scores["humanitarian"] = scores.get("humanitarian", 0) + 2

    if (any("killed" in f.lower() or "wounded" in f.lower()
            for f in signals.get("key_facts", []))
            and scores.get("conflict", 0) >= 3):
        scores["conflict"] = scores.get("conflict", 0) + 2
    # --- END Fix 2 ---

    # Find the winning type with minimum threshold of 3
    best_type = max(scores, key=scores.get) if scores else "general"
    best_score = scores.get(best_type, 0)

    if best_score < 3:
        best_type = "general"

    result = STORY_TYPES_V2[best_type]
    logger.info(f"Story type detected: {result['name']} (score={best_score})")
    return result


# ═══════════════════════════════════════════════════════════════════
# DATELINE RESOLUTION
# ═══════════════════════════════════════════════════════════════════

def resolve_dateline(articles: List[Dict]) -> str:
    """
    Return an uppercase dateline string like ``TEHRAN, March 21``.

    Tries the LLM-based location extractor first for accuracy, then
    falls back to the Counter-based approach over source location fields.

    Args:
        articles: List of source article dicts.

    Returns:
        Dateline string in "CITY, Month Day" format.

    Example:
        >>> resolve_dateline([{"location": "Tehran", "heading": "...", "story": "..."}])
        'TEHRAN, March 22'
    """
    # Try LLM-based location extractor first (more accurate than Counter)
    try:
        combined_article = {
            "heading": articles[0].get("heading", "") if articles else "",
            "story": " ".join(a.get("story", "")[:300] for a in articles[:3]),
        }
        location, _, _ = extract_location_and_category(combined_article)
        if location and location.strip() and location.lower() not in ("unknown", "india"):
            city = location.upper().strip()
            now = datetime.now()
            return f"{city}, {now.strftime('%B')} {now.day}, {now.year}"
    except Exception:
        pass  # fall through to Counter fallback

    # Fallback: Counter over location fields
    locations = [
        a.get("location", "").strip()
        for a in articles
        if a.get("location", "").strip()
        and a.get("location", "").strip().lower() != "unknown"
    ]

    if locations:
        city = Counter(locations).most_common(1)[0][0].upper()
    else:
        city = "NEW DELHI"

    now = datetime.now()
    return f"{city}, {now.strftime('%B')} {now.day}, {now.year}"


# ═══════════════════════════════════════════════════════════════════
# MAIN PROMPT BUILDER — V2
# ═══════════════════════════════════════════════════════════════════

def build_synthesis_prompt_v2(
    articles: List[Dict],
    topic: str,
    signals: Dict,
    story_type_config: Dict,
    target_words: int = 800,
    include_facts_snapshot: bool = True,
) -> Tuple[str, str, str, str]:
    # DEPRECATED — replaced by two-stage audit approach. Do not call this function.
    """
    Build a 10-section editorial prompt for article synthesis.

    Injects pre-extracted signals (quotes, key facts, human angle) into
    the prompt sections that need them, rather than dumping raw text.

    Args:
        articles:           List of source article dicts.
        topic:              Trend topic string.
        signals:            Dict from extract_article_signals().
        story_type_config:  Dict from detect_story_type_v2().
        target_words:       Minimum word count target.
        include_facts_snapshot: Whether to include key-facts snapshot section.

    Returns:
        Tuple of (system_message, user_prompt, dateline, story_type).

    Example:
        >>> sys_msg, prompt, dateline, story_type = build_synthesis_prompt_v2(
        ...     articles, "Iran protests", signals, config, 800, True)
    """
    dateline = resolve_dateline(articles)
    story_type = story_type_config.get("name", "general")
    source_digest = signals["source_digest"]

    # Story type sections — used in the narrative section prompt
    story_sections = story_type_config.get("sections", [
        "What Happened",
        "The Stakes",
        "Who Is Affected",
        "Context and Analysis",
        "Broader Implications",
        "What Happens Next",
    ])

    # Build section headings block for Phase 3
    section_headings = "\n\n".join(
        f"## {s}\n[paragraph]" for s in story_sections[:-1]
    )

    # Only add the hardcoded "Looking Ahead" closing section when it
    # is NOT already one of the story-type sections (e.g. the
    # "general" type already has "Looking Ahead" as sections[4]).
    if "Looking Ahead" not in story_sections:
        looking_ahead_block = """
# Removing the canned fallback sentence forces the model to write
# what it knows rather than copy a hedge phrase into the article.
## Looking Ahead
[Expand WHAT COMES NEXT from your plan using only confirmed
audit material. Name the upcoming decision, deadline, or open
question. Write what IS known — who is watching, what the
outcome depends on. If sources are thin, write two sentences
on the confirmed open question and stop.
Do NOT write any sentence explaining that source material is
insufficient or that implications cannot be assessed.
Never write meta-commentary about what you do not have.]"""
    else:
        looking_ahead_block = ""

    # Determine the instruction body for the position-4 section.
    # When the story type already lists "Looking Ahead" at position 4
    # (e.g. the "general" type), we must render the forward-looking
    # instruction inline so the header and instruction always match.
    # For every other story type, position 4 is a broader-implications
    # section, so we keep the original broader-implications instruction.
    _section4_name = story_sections[4].lower() if len(story_sections) > 4 else ""
    if _section4_name == "looking ahead":
        section4_instruction = (
            "[Expand WHAT COMES NEXT from your plan using only confirmed\n"
            "audit material. Name the upcoming decision, deadline, or open\n"
            "question. Write what IS known — who is watching, what the\n"
            "outcome depends on. If sources are thin, write two sentences\n"
            "on the confirmed open question and stop.\n"
            "Do NOT write any sentence explaining that source material is\n"
            "insufficient or that implications cannot be assessed.\n"
            "Never write meta-commentary about what you do not have.]"
        )
    else:
        section4_instruction = (
            "[Connect the immediate events to their wider significance —\n"
            "regional stability, institutional credibility, or precedent-setting\n"
            "consequences. Ground every claim in your audit material.\n"
            "End with a sentence that sets up the closing forward-looking section.]"
        )

    user_prompt = f"""You are about to write a news article about: {topic}

You have {len(articles)} source(s) to work from.
Read every source carefully before doing anything else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE MATERIAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{source_digest}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — AUDIT YOUR SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before writing anything else, complete this audit.
Write it inside <audit> tags exactly as shown.
Be brutally honest about what is absent — this audit
is what protects you from fabricating to fill gaps.

<audit>
MOST NEWSWORTHY FACT:
[The single most important thing that happened, in one sentence,
drawn only from the sources above. If you cannot identify one
clear fact, write "sources too thin to identify a lead fact".]

NAMED PEOPLE:
[Every person named in the sources. For each, write their exact
stated role if the source gives one. If the source gives no role,
write "no role stated". Do not use your memory to add a title.]

DIRECT QUOTES:
[Copy any text inside quotation marks from the sources, verbatim.
If none exist, write "none".]

KEY NUMBERS AND DATES:
[Every figure, percentage, count, monetary amount, and date that
appears explicitly in the sources. If none, write "none".]

NAMED INSTITUTIONS:
[Every organisation, government body, country, or official body
named in the sources. If none, write "none".]

WHAT I DO NOT HAVE:
[Facts a reader would reasonably expect that are absent from the
sources. Be specific. Example: "No casualty figures", "No official
government response", "No timeline of events". This section
defines the limits of what you are permitted to write.]
</audit>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — PLAN YOUR STORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using only what appeared in your audit, plan the article
as a continuous story. Write it inside <plan> tags.
One sentence per movement. This map governs everything
you write in Phase 3.

<plan>
OPENING HOOK:
[The one fact or moment that pulls the reader in.
This becomes the first sentence of your lead paragraph.]

CENTRAL TENSION:
[What makes this story not simple — the competing interest,
the unanswered question, or the stakes. Every news story
has one. Name it specifically using your audit material.]

HUMAN DIMENSION:
[Who bears the consequence of these events and how.
If your audit's NAMED PEOPLE section is empty or your
sources contain no human impact detail, write:
"sources do not contain human impact — will note honestly."]

BROADER PICTURE:
[Why this matters beyond the immediate story. Name one
country or institution from your audit's NAMED INSTITUTIONS.
If NAMED INSTITUTIONS is "none", write:
"sources do not support broader implications section —
will state this honestly rather than invent."]

# Removing the literal hedge phrase stops it propagating into
# the article body through the plan. Redirect to a real question.
WHAT COMES NEXT:
[One specific upcoming event, decision, or deadline confirmed
in your sources. If no future event is confirmed, name the
single open question the story leaves unresolved — write it
as a direct question, not a hedge sentence.
Example: "Will Liverpool replace Salah before the window?"
Do NOT write the phrase "no confirmed next event in sources"
— that is a planning note, not plan content.]

NARRATIVE ARC:
[One sentence describing the shape of the whole story.
Example: "This story moves from policy announcement to
economic uncertainty to unresolved geopolitical tension."
This sentence keeps your sections connected as you write.]
</plan>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — WRITE THE ARTICLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have your audit. You have your plan. Now write.

Every fact must trace to your audit.
Every section must serve the narrative arc in your plan.
The sections are movements in a single story — write them
as such, not as isolated boxes to fill.

FLOW RULE — apply to every section transition:
The last sentence of each section must do one of two things:
  (a) Answer a question and raise a new one the next section
      will address — pulling the reader forward naturally.
  (b) State a consequence or tension the next section will
      explore — creating continuity, not a hard stop.
Read your last sentence before moving to the next section.
If it could be the last sentence of an unrelated article,
rewrite it until it cannot.

ANTI-HALLUCINATION RULE:
You may only use what is in your audit under NAMED PEOPLE,
DIRECT QUOTES, KEY NUMBERS AND DATES, and NAMED INSTITUTIONS.
You may not use anything from WHAT I DO NOT HAVE.
If a section cannot be filled honestly from your audit,
write one sentence saying what is not yet known, then move on.

ATTRIBUTION RULE:
Attribute only to people and institutions in your audit.
Write their exact stated role if your audit contains it.
If your audit says "no role stated", write their name only.
Never add a title your audit does not contain.
If your audit's DIRECT QUOTES is "none", use reported
speech only: "[Name] said that..." not "[Name] stated '...'".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTICLE OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write your article now, starting immediately below.
The very first line must be the # headline.
The very second line must be the ### subheadline.
Nothing before the headline. Nothing between headline
and subheadline.

# [Headline — 10–15 words, active voice, names who did what, about {topic}]
### [Subheadline — one sentence, max 150 characters, adds context not already in the headline]

{dateline} —

[Lead paragraph: expand your OPENING HOOK into 2–3 sentences.
Answer who, what, where, when. Make the reader need the next paragraph.
Do not summarise the whole article here — just earn the next sentence.]

# Removing the explicit fallback instruction prevents the model
# from copying a meta-commentary sentence into the published list.
**Key facts:**
- [fact drawn from audit]
- [fact drawn from audit]
- [fact drawn from audit]
- [fact drawn from audit — write only as many bullets as you have
  confirmed facts. If you have fewer than 4 stop at 3 or 2.
  Do NOT write a bullet explaining that facts are unavailable.
  Silence is better than scaffolding.]

## {story_sections[0] if len(story_sections) > 0 else "What Happened"}
[Expand OPENING HOOK and CENTRAL TENSION from your plan.
End with a sentence that introduces the human dimension
or raises the stakes — bridging naturally to the next section.]

## {story_sections[1] if len(story_sections) > 1 else "The Stakes"}
[Expand CENTRAL TENSION. Why does this matter?
Who or what is at risk? Ground it in your audit material.
End with a sentence that brings in the human or broader dimension.]

## {story_sections[2] if len(story_sections) > 2 else "Who Is Affected"}
[Expand HUMAN DIMENSION from your plan.
If your plan says sources do not contain human impact, write:
"The direct human consequences of [specific event from audit]
are not yet clear from available reporting. What is confirmed
is [one fact from audit]."
End with a sentence that opens toward the broader picture.]

## {story_sections[3] if len(story_sections) > 3 else "Context and Analysis"}
[Historical background that illuminates the current situation.
When you move from fact to interpretation, write "Analysis:" before
that sentence so the reader knows. End with a sentence that raises
the implications for the region or world.]

## {story_sections[4] if len(story_sections) > 4 else "Broader Implications"}
{section4_instruction}

{looking_ahead_block}

Minimum {target_words} words. AP Style throughout.
This is journalism, not a form. Write it as a story.
"""

    return SYSTEM_MESSAGE_V3, user_prompt, dateline, story_type


# ═══════════════════════════════════════════════════════════════════
# POST-GENERATION VALIDATOR — V2
# ═══════════════════════════════════════════════════════════════════

def validate_article_v2(article: Dict, topic: str, signals: Dict,
                        story_type_config: dict = None) -> Dict:
    # DEPRECATED — replaced by two-stage audit approach. Do not call this function.
    """
    Hard post-generation validator for 10-section articles.

    Checks required sections, quote presence, word count, topic drift,
    and outlet-name leaks. Returns a result dict that the caller uses
    to decide whether to retry or proceed.

    Args:
        article: Parsed article dict with 'heading', 'sub_heading', 'story'.
        topic:   Trend topic string.
        signals: Dict from extract_article_signals() (used for context).
        story_type_config: Optional story type config dict from detect_story_type_v2().

    Returns:
        dict with keys:
            passes   – bool (True only if failures list is empty)
            failures – list of failure-reason strings (block upload)
            warnings – list of warning strings (logged but do not block)

    Example:
        >>> result = validate_article_v2(article, "Iran protests", signals)
        >>> if not result["passes"]:
        ...     print(result["failures"])
    """
    story = article.get("story", "")
    heading = article.get("heading", "")
    failures: List[str] = []
    warnings: List[str] = []

    # --- Fix 1 — structural format checks ---
    heading_words = heading.strip().split()
    if len(heading_words) < 4:
        failures.append(
            "Section 1 — headline missing or malformed: model did not produce a "
            "# headline line (got: '{}')".format(heading.strip()[:80])
        )

    sub_heading = article.get("sub_heading", "")
    if not sub_heading.strip():
        failures.append(
            "Section 2 — subheadline missing: model did not produce a "
            "### subheadline line"
        )
    # --- END Fix 1 ---

    # ── Section 4: Key facts snapshot ──
    if not re.search(r'\*\*Key facts\*\*|key facts:', story, re.IGNORECASE):
        failures.append("Missing required section: Key Facts snapshot")

    # ── Dynamic section validation ──
    # Check for sections that the prompt actually told the model to write,
    # not a hardcoded list from the previous prompt architecture.
    # story_type_config["sections"] contains the sections used to build
    # the prompt for this specific article generation attempt.
    story_type_sections = (
        story_type_config.get("sections", [])
        if story_type_config
        else []
    )

    # These two sections are always required regardless of story type
    # because the three-phase prompt always includes them
    # "What Happens Next" removed from prompt — "Looking Ahead" is
    # now the sole canonical forward-looking section to validate.
    always_required = ["Looking Ahead"]

    all_required = story_type_sections + [
        s for s in always_required if s not in story_type_sections
    ]

    for section in all_required:
        if f"## {section}" not in story:
            failures.append(f"Missing required section: {section}")

    # ── Quote presence check ──
    # Check for (a) actual quote characters with 15+ char content, or
    # (b) attribution phrases indicating reported speech
    has_quotes = bool(
        re.search(r'[\u201c\u201d"""][^"\u201c\u201d\u201e]{15,}[\u201c\u201d"""]', story)
    )
    has_attribution = bool(
        re.search(
            r'\b(?:stated|confirmed|told reporters|said in a statement)\b',
            story,
            re.IGNORECASE,
        )
    )
    if not has_quotes and not has_attribution:
        failures.append(
            "No quotes or attributed statements found — article needs ≥2 "
            "named quotes or attribution phrases"
        )

    # ── Word count check ──
    word_count = len(story.split())
    if word_count < 700:
        failures.append(f"Word count too low: {word_count} (minimum 700)")

    # ── Topic drift check (warning, not failure) ──
    # Extract 4+ char keywords from topic for checking
    topic_keywords = [w.lower() for w in topic.split() if len(w) >= 4]
    if topic_keywords:
        paragraphs = story.split("\n\n")
        off_topic_count = 0
        for para in paragraphs:
            para_stripped = para.strip()
            # Skip section headers and short paragraphs
            if para_stripped.startswith("##") or len(para_stripped) < 50:
                continue
            para_lower = para_stripped.lower()
            if not any(kw in para_lower for kw in topic_keywords):
                off_topic_count += 1
        if off_topic_count > 2:
            warnings.append(
                f"Possible topic drift: {off_topic_count} paragraphs have "
                f"no overlap with topic keywords ({', '.join(topic_keywords[:5])})"
            )

    # ── Outlet name leak check (warning, not failure) ──
    outlet_matches = _OUTLET_LEAK_RE.findall(story)
    if outlet_matches:
        unique_outlets = list(set(outlet_matches))
        warnings.append(
            f"Outlet names found in body text: {', '.join(unique_outlets)}"
        )

    # --- Fix 3 — Section 5 heading match check ---
    if story_type_config and story_type_config.get("sections"):
        # Extract all ## headers from the body
        all_headers = re.findall(r'^## (.+)$', story, re.MULTILINE)

        # Fixed-section keywords to exclude (sections 7-10 headers)
        fixed_keywords = {
            "context", "analysis", "broader", "implications", "regional",
            "numbers", "data", "timeline", "what", "looking"
        }

        # Filter to Section 5 narrative headers only
        narrative_headers = [
            h for h in all_headers
            if not any(kw in h.lower() for kw in fixed_keywords)
        ]

        if not narrative_headers:
            warnings.append(
                "Section 5 — no ## section headings found in narrative body"
            )
        else:
            # Build vocabulary from expected section headings
            expected_vocab = set()
            for section_heading in story_type_config["sections"]:
                for word in section_heading.lower().split():
                    if len(word) >= 4:
                        expected_vocab.add(word)

            # Score each narrative header against expected vocabulary
            matched = [
                h for h in narrative_headers
                if any(w in expected_vocab
                       for w in h.lower().split() if len(w) >= 4)
            ]

            if len(matched) == 0:
                warnings.append(
                    "Section 5 — narrative headings do not match detected story "
                    "type '{}': found {}, expected vocabulary from {}".format(
                        story_type_config.get("name", "unknown"),
                        narrative_headers,
                        story_type_config["sections"]
                    )
                )
    # --- END Fix 3 ---

    return {
        "passes": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════
# DYNAMIC PROMPT BUILDER — TWO-STAGE AUDIT APPROACH
# ═══════════════════════════════════════════════════════════════════


def build_dynamic_prompt(
    articles: List[Dict],
    topic: str,
    audit: AuditResult,
    signals: Dict,
) -> Tuple[str, str]:
    """
    Build a dynamic prompt driven by the Stage 1 audit result.

    Generates 7-section article prompts where each section's depth
    scales to what the source material can honestly support.
    Returns (system_message, user_prompt).

    Args:
        articles: List of source article dicts.
        topic:    Trend topic string.
        audit:    AuditResult from audit_source_material().
        signals:  Dict from extract_article_signals().

    Returns:
        Tuple of (system_message, user_prompt).
    """
    now = datetime.now()
    city = (audit.primary_location or "INTERNATIONAL").upper()
    dateline = f"{city}, {now.strftime('%B')} {now.day}, {now.year}"
    source_digest = signals.get("source_digest", "")

    # ── Build quote_block: structured, regex-extracted quotes ──
    # Quotes remain in the source digest for context; this block surfaces them
    # prominently BEFORE SOURCE MATERIAL so the model reads what it can quote
    # verbatim before it encounters the full raw context.
    quotes = signals.get("quotes", [])
    if quotes and audit.has_direct_quotes:
        lines = []
        for i, q in enumerate(quotes, 1):
            speaker = q.get("speaker", "unnamed official")
            text = q.get("text", "").strip()
            if text:
                lines.append(f'{i}. "{text}" — {speaker}')
        if lines:
            quote_block = (
                "DIRECT QUOTES EXTRACTED FROM SOURCES — "
                "use these verbatim in the article body:\n"
                + "\n".join(lines)
                + "\n\nRule: After each quote you use, write exactly one "
                "sentence explaining why that speaker said it at that "
                "specific moment — not what they said, but what they "
                "were trying to achieve or signal.\n"
                "Use the attribution phrase provided with each quote exactly "
                "as written. Do not replace it with \"unnamed official.\" If "
                "the attribution says \"Tehran's central command stated,\" "
                "write that. Preserve the sourcing language from the "
                "original reporting.\n\n"
            )
        else:
            quote_block = ""
    else:
        quote_block = ""

    # ── Build raw_quotes_block: sentence-level fallback for quotes the regex missed ──
    # Catches quote structures where the speaker is identified in a prior sentence
    # (e.g. "Koch said.\n'We had a collective expression of joy.'") that _QUOTE_RE
    # cannot capture because the capitalised name does not follow the closing quote.
    raw_quote_sentences = []
    seen_texts = {q.get("text", "")[:40] for q in quotes}
    for article in articles[:6]:
        source_name = article.get("source_name", "Unknown")
        story = article.get("story", "")
        sentences = re.split(r'(?<=[.!?])\s+', story)
        for sent in sentences:
            has_quote_char = (
                '"' in sent or
                '\u201c' in sent or
                '\u2018' in sent
            )
            if has_quote_char and len(sent) > 40:
                fingerprint = sent.strip()[:40]
                if fingerprint not in seen_texts:
                    seen_texts.add(fingerprint)
                    raw_quote_sentences.append(
                        f"  [{source_name}]: {sent.strip()}"
                    )
    if raw_quote_sentences and audit.has_direct_quotes:
        raw_quotes_block = (
            "ADDITIONAL QUOTED SENTENCES FROM SOURCES "
            "(regex extraction may have missed these — "
            "use any direct speech found here):\n"
            + "\n".join(raw_quote_sentences[:8])
            + "\n\n"
        )
    else:
        raw_quotes_block = ""

    is_rich = audit.source_quality == "rich"
    is_thin = audit.source_quality == "thin"

    # Scale the number of required headers by source quality.
    # Defined here (before system_message) to fix ordering: constraint_header
    # uses header_count and must be assigned before system_message at line 1193.
    if is_rich:
        header_rule = "Use 4-5 ## section headers to divide the body (one before each major section)."
    elif is_thin:
        header_rule = "Use at most 1 ## section header, or none if the article is under 300 words."
    else:
        header_rule = "Use 2-3 ## section headers to divide the body."

    # Numeric header count used in constraint blocks — must stay in sync with header_rule above
    header_count = 5 if is_rich else (1 if is_thin else 3)

    constraint_header = (
        "NON-NEGOTIABLE LIMITS — enforced before any instruction below:\n"
        f"1. Word ceiling: {audit.honest_word_ceiling}. "
        f"   Stop writing when you reach {audit.honest_word_ceiling} words. "
        "   This is not a target. Exceeding it is a hard failure.\n"
        f"2. Section header (##) limit: {header_count}. "
        f"   Writing more than {header_count} ## headers is a hard failure.\n"
        f"3. Source quality assessed as: {audit.source_quality.upper()}. "
        "   Write to the depth this quality supports — no more.\n"
        f"4. Direct quotes confirmed in sources: {audit.has_direct_quotes}. "
        + ("   Use them verbatim.\n" if audit.has_direct_quotes
           else "   Do not fabricate or paraphrase as if quoting.\n")
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    # ── Section 3: Narrative ──
    if is_rich:
        narrative_instruction = (
            "Begin by advancing from where Lead ended — do not restate the cause, "
            "start date, or any fact already in Lead. "
            "Write 2-3 paragraphs developing the story chronologically or thematically. "
            "Weave key numbers, dates, and statistics into the prose at the exact moment "
            "they become meaningful — do not list them as bullets. "
            "After each number, write one sentence on its human consequence. "
            "If a quote from a key actor captures a pivotal moment in the chronology, "
            "embed it here with one sentence on why that person said it at that moment. "
            "End with the central unresolved tension."
        )
    elif is_thin:
        narrative_instruction = (
            "Begin by advancing from where Lead ended — do not restate any fact "
            "already in Lead. Write 1 paragraph expanding on what is confirmed in sources. "
            "Weave any numbers or dates into the prose at the moment they become meaningful — "
            "do not list them as bullets. After each number, write one sentence on its consequence. "
            "Do not speculate beyond what is explicitly stated."
        )
    else:
        narrative_instruction = (
            "Begin by advancing from where Lead ended — do not restate any fact "
            "already in Lead. "
            "Write 1-2 paragraphs developing confirmed facts. "
            "Weave key numbers, dates, and statistics into the prose at the moment "
            "they become meaningful — do not list them as bullets. "
            "After each number, write one sentence on its consequence. "
            "Show cause and consequence where your sources support it."
        )

    # ── Section 4: Voices ──
    if audit.has_direct_quotes:
        voices_instruction = (
            "Open with a one-clause bridge connecting to what just happened in What Happened. "
            "Use quotes not already embedded in other sections. Introduce each speaker "
            "with their name and role. After EACH quote, write exactly one sentence "
            "explaining what the speaker was trying to achieve — not what they said, "
            "but WHY they said it at this specific moment, and what it cost or gained them. "
            "If all direct quotes were already used in earlier sections, write attributed "
            "paraphrases of secondary voices using 'said' or 'told reporters'."
        )
    elif audit.has_named_sources:
        voices_instruction = (
            "Open with a one-clause bridge connecting to what just happened in What Happened. "
            "No direct quotes are available. Write attributed paraphrases "
            "using 'said', 'stated', or 'told reporters'. Introduce each "
            "with the speaker's name. Do not use quotation marks. "
            "After each paraphrase, write one sentence on why this person spoke at this moment."
        )
    else:
        voices_instruction = (
            "No named sources are available. Write one sentence noting what "
            "official response or statement, if any, is absent from reporting."
        )

    # ── Section 5: Analysis & Context ──
    if audit.has_expert_opinion and is_rich:
        analysis_instruction = (
            "Begin with a cause-and-effect bridge from What Happened. "
            "Write 2 paragraphs. First: historical or policy context explaining "
            "the specific trigger for this event — not general background. "
            "If a statistic proves your point, embed it here and follow it with "
            "one sentence on its human consequence. "
            "Second: label it 'Analysis:' and write one paragraph of reasoned "
            "interpretation grounded in source material. "
            "If expert opinion was not already used in Voices, embed one attributed "
            "quote or paraphrase here, followed by one sentence on what it reveals "
            "about the broader significance. "
            "Do not repeat any statistic or quote already used in prior sections."
        )
    else:
        analysis_instruction = (
            "Begin with a cause-and-effect bridge from What Happened. "
            "Write 1 paragraph of context explaining the specific trigger for this event, "
            "using only what your sources explicitly state. "
            "If a statistic or attributed statement proves the context, embed it here "
            "and follow it immediately with its human consequence. "
            "Do not repeat any fact, statistic, or quote already stated above."
        )

    # ── Section 6: Implications ──
    if audit.has_impact_data and not is_thin:
        implications_instruction = (
            "Name a specific country, institution, or market not yet mentioned in "
            "Analysis & Context. Do not repeat any consequence already stated above. "
            "Write 1-2 paragraphs on significance. "
            "Connect to one measurable consequence — price, policy, population — "
            "that traces directly to what the source material states."
        )
    else:
        implications_instruction = (
            "Write one sentence identifying the single most significant "
            "consequence of this event beyond its immediate location, "
            "naming a specific country, market, or institution. "
            "Use only what sources state — do not repeat what Analysis & Context said."
        )

    # ── Section 7: What's Next ──
    if audit.has_future_event:
        next_instruction = (
            "Open with a one-clause bridge from Implications — what does that "
            "consequence mean for what comes next? "
            "Then write 2-3 sentences on confirmed upcoming events, deadlines, "
            "or decisions from your sources. Name dates and decision-makers."
        )
    else:
        next_instruction = (
            "Open with a one-clause bridge from Implications — what does that "
            "consequence mean for what comes next? "
            "Then write one sentence posing the single open question this story "
            "leaves unresolved. Write it as a direct question, not a hedge."
        )

    word_floor = {"rich": 600, "adequate": 350, "thin": 180}.get(
        audit.source_quality, 250
    )
    word_ceiling = audit.honest_word_ceiling

    quality_warning = (
        "\n⚠️ SOURCE WARNING: Thin sources. Write fewer words honestly "
        "rather than more words with fabrication.\n"
        if is_thin else ""
    )

    system_message = constraint_header + (
        "You are a senior wire service journalist with twenty years of "
        "field reporting experience writing for an educated general audience.\n\n"
        "BEFORE WRITING — identify silently:\n"
        "1. The single human tension at the center of this story.\n"
        "2. The emotional arc: what changes from the opening paragraph to the final one "
        "(e.g. 'isolation → coalition → open question').\n"
        "3. Which paragraph carries the most narrative weight for this particular story.\n"
        "Write from those answers. Do not write from the blueprint template.\n\n"
        "FORMAT RULE: After the headline (# ...) and subheadline (### ...) at the very top, "
        "divide the article body into sections using ## headers. "
        "Each ## header must name the specific aspect of THIS story covered in the section "
        "that follows — never a generic blueprint label. "
        "Good examples: '## How Iran's Air Defenses Repelled the Strike', "
        "'## The Factions Tearing the ADC Apart', "
        "'## Why the Strait of Hormuz Matters to Every Economy'. "
        "FORBIDDEN generic headers — hard failure if any appear: "
        "## Lead, ## Background, ## What Happened, ## Voices, ## Analysis & Context, "
        "## Implications, ## What's Next, ## Key Facts. "
        "No ③ symbols, no bold section titles. "
        "The writing blueprint is guidance only — its section labels must never appear in output.\n\n"
        "STORY RULE: A fact without explanation is data, not journalism. "
        "Every number, quote, and named claim must be followed immediately by its human meaning — "
        "why this specific fact matters to a specific person or group right now. "
        "Never drop a statistic and move on. Never end a quote without saying what it cost "
        "or gained the person who said it.\n\n"
        "SCATTER RULE: Place each fact and quote at the paragraph in the story where it lands hardest. "
        "A quote that illuminates a cause belongs in the development paragraphs. "
        "A statistic that proves consequence belongs in the stakes paragraph. "
        "A number that supports analysis belongs in the context paragraph. "
        "Do not cluster all quotes together or all numbers together.\n\n"
        "ATTRIBUTION RULE: Attribute every contested or significant claim to its source: "
        "'According to [institution]...' or '[Name] told reporters...'. "
        "Never state a disputed fact as if it were settled.\n\n"
        "OPENING RULE: The first paragraph answers Who, What, Where, When, and Why "
        "in 2-3 sentences using the single most newsworthy confirmed fact as the opening clause. "
        "It does not summarise the whole story — it earns the next paragraph.\n\n"
        "PAIRING RULES — apply everywhere:\n"
        "Stat rule: immediately after any number write one sentence on its specific human consequence.\n"
        "Quote rule: immediately after any quote write one sentence on why the speaker said it at "
        "this specific moment and what it cost or gained them.\n"
        "  BAD (forbidden): 'She said this to emphasize the need for a "
        "ceasefire.' — this restates the quote's topic, names no specific "
        "consequence, and could follow any quote in any article.\n"
        "  GOOD: 'Svyrydenko named Russia's tactics publicly because EU "
        "defence ministers were meeting in Brussels the next morning and "
        "she needed their attention before the session opened.' — this names "
        "a specific actor, a specific moment, and a specific strategic "
        "consequence that is unique to this quote at this time.\n"
        "For unnamed officials specifically: do not write "
        "'The official said this to emphasize X.' "
        "Instead name the institutional consequence — which government, "
        "which policy, which deadline their statement was directed at.\n"
        "  BAD: 'The official said this to highlight the severity of "
        "the situation.'\n"
        "  GOOD: 'The statement came as the UN Security Council was "
        "scheduled to vote on a resolution within 48 hours, making "
        "public pressure the only remaining lever.'\n"
        "Cause rule: every cause you name must be followed by its effect in the next sentence.\n\n"
        "VARIATION RULE: Before writing each paragraph, look at the first word of the previous "
        "paragraph. Your new paragraph must open with a different subject, actor, or angle. "
        "Never start two consecutive paragraphs with the same word or phrase. "
        "Vary sentence structure, tense, and point of entry across every paragraph.\n\n"
        "NO-REPETITION RULE: Each paragraph must introduce confirmed information — a fact, quote, "
        "timeline element, or perspective — not present anywhere earlier in the article. "
        "Before writing each paragraph ask: 'Does this tell the reader something they cannot "
        "already infer from what I have written?' If not, delete the paragraph and move to the "
        "next section. Stop writing when the story is complete; do not pad to reach a word count.\n\n"
        "BANNED PHRASES — hard failure if any appear:\n"
        "'as the situation continues', 'increasingly important', "
        "'further complicated', 'regional and global consequences', "
        "'it is becoming clear', 'the situation deteriorates', "
        "'it is worth noting', 'it remains to be seen', "
        "'in a significant development', 'amid growing concerns', "
        "'has sparked debate', 'raises questions about', "
        "'underscored the importance', 'highlighted the need', "
        "'at a critical juncture', 'crucial', 'landmark', 'historic', "
        "'unprecedented', 'sparking', ' amid '.\n\n"
        "If you are about to write a banned phrase: stop, delete it, "
        "write a specific verified fact from sources instead."
    )

    dev_count = "2-3 paragraphs" if is_rich else ("1 paragraph" if is_thin else "1-2 paragraphs")
    voices_count = "1-2 paragraphs" if audit.has_direct_quotes else "1 paragraph"
    context_count = "2 paragraphs" if (audit.has_expert_opinion and is_rich) else "1 paragraph"
    stakes_count = "1-2 paragraphs" if (audit.has_impact_data and not is_thin) else "1 sentence"

    user_prompt = f"""Write a news story about: {topic}
{quality_warning}
{quote_block}{raw_quotes_block}SOURCE MATERIAL:
{source_digest}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# [Headline — active voice, names who did what, 10-15 words]
### [Subheadline — one sentence, max 150 characters, adds context not in headline]

{dateline}

[Lead paragraph — no header above it]

## [Story-specific header naming the exact angle of the next section]

[Remaining story sections, each preceded by a ## header]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WRITING BLUEPRINT — instructions only, do NOT copy labels into output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① OPENING (2-3 sentences — NO header above this section)
Write 2-3 sentences answering Who, What, Where, When, Why.
Use the single most newsworthy confirmed fact as your opening clause.
Do not summarise the whole story. Earn the next paragraph.

② DEVELOPMENT ({dev_count})
Before this section write a ## header that names the specific development angle
(e.g. "## How the Strike Unfolded" or "## The Chain of Events That Led Here").
{narrative_instruction}

③ VOICES ({voices_count})
Before this section write a ## header that names whose voices dominate
(e.g. "## What Officials and Witnesses Said" or "## The ADC Responds").
{voices_instruction}

④ CONTEXT ({context_count})
Before this section write a ## header that frames the historical or analytical lens
(e.g. "## A Rivalry Decades in the Making" or "## Nigeria's Electoral Fault Lines").
{analysis_instruction}

⑤ STAKES ({stakes_count})
Before this section write a ## header that names what is concretely at risk
(e.g. "## The Economic Cost Already Being Felt" or "## What Failure Would Mean").
{implications_instruction}

⑥ HORIZON (1 paragraph)
Before this section write a ## header that signals what comes next
(e.g. "## The Votes and Decisions Still to Come" or "## What the Crew Does Next").
{next_instruction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Target {word_floor}–{word_ceiling} words. Hard ceiling: stop at {word_ceiling} words.
  Do not add paragraphs to reach the floor if the story is already complete.
- Every fact must trace directly to the source material above
- All six story elements must appear — scale depth, never skip
- Each paragraph must introduce new information not stated earlier in the article.
  Repeating a fact in different words is a writing error — delete and move on.
- No two consecutive paragraphs may start with the same word or phrase.
- {header_rule} Headers must be story-specific — never generic blueprint labels.
- No banned phrases
- AP Style throughout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL REMINDER — CHECK BEFORE YOUR FIRST WORD:
• Ceiling: {audit.honest_word_ceiling} words maximum. Count as you write. Stop when you reach it.
• Headers: {header_count} ## header(s) maximum.
• Banned word check: 'historic', 'unprecedented', 'crucial', 'landmark' must not appear anywhere in output.
• Quotes: {audit.has_direct_quotes}. {"Direct quotes are available above. Use them." if audit.has_direct_quotes else "No quotes available. Do not invent any."}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write now — output the headline, subheadline, dateline, then the story:"""

    system_message += (
        "\n\nFINAL CHECK BEFORE EACH PARAGRAPH: Read the first word of "
        "the paragraph you just wrote. Your next paragraph MUST begin "
        "with a different grammatical subject — a name, a number, a "
        "country, a quote, a cause. If your last paragraph started with "
        "'The', your next must not. Violating this is a hard failure."
    )

    return system_message, user_prompt


# ═══════════════════════════════════════════════════════════════════
# DYNAMIC VALIDATOR — TWO-STAGE AUDIT APPROACH
# ═══════════════════════════════════════════════════════════════════


def validate_article_dynamic(article: Dict, audit: AuditResult) -> Dict:
    """
    Validate a generated article against the audit result.

    Enforces banned phrases as hard failures that trigger retry.

    Args:
        article: Parsed article dict with 'heading', 'story'.
        audit:   AuditResult from audit_source_material().

    Returns:
        dict with keys:
            passes   – bool (True only if failures list is empty)
            failures – list of failure-reason strings
            warnings – list of warning strings
    """
    failures: List[str] = []
    warnings: List[str] = []

    # Check 1 — Minimum word count
    word_count = len(article.get("story", "").split())
    if word_count < 150:
        failures.append(f"Article too short: {word_count} words (minimum 150)")

    # Check 1b — Maximum word count (hard ceiling from audit)
    ceiling_limit = audit.honest_word_ceiling + 50
    if word_count > ceiling_limit:
        failures.append(
            f"Word ceiling exceeded: {word_count} words written, "
            f"ceiling is {audit.honest_word_ceiling} "
            f"(+50 tolerance = {ceiling_limit}). "
            f"Trim the least essential paragraph and regenerate."
        )

    # Check 2 — Headline present
    if len(article.get("heading", "").strip().split()) < 4:
        failures.append("Headline missing or too short")

    story = article.get("story", "")

    # Check 1c — Forbidden generic section headers (hard failure)
    # Headers must name the specific angle of THIS story. The following
    # generic category labels are forbidden regardless of capitalisation.
    found_headers = re.findall(r'^## (.+)$', story, re.MULTILINE)
    for h in found_headers:
        if h.strip().lower() in _FORBIDDEN_HEADERS:
            failures.append(
                f"Forbidden generic section header: '## {h.strip()}'. "
                f"Write a story-specific header naming the exact angle "
                f"of this section — never a category label."
            )

    # Check 3 — Accidental blueprint section headers (hard failure)
    # These exact blueprint labels must never appear; they indicate the model
    # echoed the writing template instead of writing story-specific headers.
    ACCIDENTAL_HEADERS = [
        "## Lead",
        "## What Happened",
        "## Voices",
        "## Analysis & Context",
        "## Implications",
        "## What's Next",
        "## Key Facts",
        "## Background",
    ]
    for h in ACCIDENTAL_HEADERS:
        if h in story:
            failures.append(
                f"Accidental section header in prose: '{h}' — FORMAT RULE violated"
            )

    # Check 3b — Generic non-story-specific headers (hard failure)
    # The model is instructed to write headers that name the specific angle of
    # THIS story. The following generic labels are forbidden regardless of
    # whether they match the blueprint exactly.
    GENERIC_HEADERS = {
        "## The Human Cost",
        "## What Comes Next",
        "## The Background",
        "## The Stakes",
        "## The Context",
        "## The Response",
        "## The Fallout",
        "## The Aftermath",
        "## The Impact",
        "## Analysis",
        "## Context",
        "## The Analysis",
        "## The Implications",
        "## Response",
        "## What This Means",
        "## Moving Forward",
        "## Going Forward",
        "## Looking Ahead",
        "## The Bigger Picture",
        "## The Bottom Line",
    }
    for line in story.splitlines():
        stripped_line = line.strip()
        if stripped_line in GENERIC_HEADERS:
            failures.append(
                f"Generic section header detected: '{stripped_line}' — "
                "headers must name the specific story angle, not a generic label"
            )

    # Check 3c — Short (≤3 word) ## headers (warning — likely too generic)
    for line in story.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("## "):
            header_words = stripped_line[3:].strip().split()
            if len(header_words) <= 3:
                warnings.append(
                    f"Section header may be too generic (≤3 words): '{stripped_line}' — "
                    "headers should name the specific story angle"
                )

    # Check 4 — Quote presence when audit indicates quotes available
    if audit.has_direct_quotes:
        has_quotes = bool(
            re.search(
                r'[\u201c\u201d"\u2018\u2019][^"\u201c\u201d\u201e]{15,}[\u201c\u201d"\u2018\u2019]',
                article.get("story", ""),
            )
        )
        if not has_quotes:
            warnings.append(
                "Audit indicated direct quotes available but none found in output"
            )

    # Check 5 — Banned phrases (HARD FAILURE — triggers retry)
    BANNED_PHRASES = [
        "as the situation continues",
        "increasingly important",
        "further complicated",
        "regional and global consequences",
        "it is becoming clear",
        "the situation deteriorates",
        "it is worth noting",
        "it remains to be seen",
        "in a significant development",
        "amid growing concerns",
        "has sparked debate",
        "raises questions about",
        "underscored the importance",
        "highlighted the need",
        "at a critical juncture",
        "crucial",
        "landmark",
        "historic",
        "unprecedented",
        "sparking",
        " amid ",
    ]

    for phrase in BANNED_PHRASES:
        if phrase.lower() in story.lower():
            failures.append(f"Banned phrase detected: '{phrase}'")

    # Check 5 — Fabrication markers (WARNING only, not failure)
    fabrication_markers = [
        "not yet clear from available reporting",
        "sources do not contain",
        "no confirmed",
        "cannot be determined",
    ]
    found_markers = [m for m in fabrication_markers if m in story.lower()]
    if found_markers:
        warnings.append(
            f"Meta-commentary phrases found in article body — review: {found_markers}"
        )

    # Check 6 — Consecutive paragraph-start repetition (WARNING only)
    # Split on blank lines; skip lines that are headers (## ...) or empty.
    paragraphs = [
        p.strip() for p in story.split("\n\n")
        if p.strip() and not p.strip().startswith("#")
    ]
    if len(paragraphs) >= 3:
        for idx in range(len(paragraphs) - 1):
            words_a = paragraphs[idx].split()
            words_b = paragraphs[idx + 1].split()
            first_a = words_a[0].lower().rstrip(".,;:") if words_a else ""
            first_b = words_b[0].lower().rstrip(".,;:") if words_b else ""
            if first_a and first_a == first_b:
                warnings.append(
                    f"Consecutive paragraphs {idx + 1} and {idx + 2} both start "
                    f"with '{words_a[0]}' — VARIATION RULE may have been violated"
                )
                break  # one warning per article is sufficient

    return {"passes": len(failures) == 0, "failures": failures, "warnings": warnings}


# ═══════════════════════════════════════════════════════════════════
# ARTICLE PARSER (unchanged from V1)
# ═══════════════════════════════════════════════════════════════════

def parse_generated_article(generated_text: str) -> Dict:
    """
    Parse LLM-generated text into a structured article dict.
    Strips <audit> and <plan> blocks produced by the three-phase
    prompt before extracting heading / sub_heading / story.
    """
    if not generated_text:
        return {"heading": "", "sub_heading": "", "story": ""}

    # ── Strip Phase 1 and Phase 2 thinking blocks ──
    # The three-phase prompt produces <audit>...</audit> and
    # <plan>...</plan> before the article. Remove them so they
    # do not appear in the stored article or CMS upload.
    import re as _re
    generated_text = _re.sub(
        r"<audit>.*?</audit>",
        "",
        generated_text,
        flags=_re.DOTALL,
    ).strip()
    generated_text = _re.sub(
        r"<plan>.*?</plan>",
        "",
        generated_text,
        flags=_re.DOTALL,
    ).strip()

    # ── Existing parsing logic continues below unchanged ──
    lines = generated_text.strip().split("\n")

    # ── Extract headline (first line starting with # or ##) ──
    heading = ""
    headline_index = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            heading = stripped.lstrip("# ").strip()
            headline_index = i
            break

    # ── Extract subheading (first ### after headline) ──
    sub_heading = ""
    subheading_index = -1

    if headline_index >= 0:
        for i in range(headline_index + 1, min(headline_index + 10, len(lines))):
            stripped = lines[i].strip()
            if stripped.startswith("### "):
                sub_heading = stripped.replace("### ", "", 1).strip()
                if len(sub_heading) > 150:
                    sub_heading = sub_heading[:147] + "..."
                subheading_index = i
                break

    # ── Fallback: use first non-empty, non-heading line ──
    if not heading:
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("#"):
                heading = line.strip()
                headline_index = i
                break

    # ── Extract body ──
    body_start = max(headline_index, subheading_index)
    body_lines = lines[body_start + 1:] if body_start >= 0 else lines[1:]

    story = "\n".join(body_lines).strip()
    story = re.sub(r"^[\s\n]+", "", story)
    story = re.sub(r"[\s\n]+$", "", story)

    return {
        "heading": heading,
        "sub_heading": sub_heading,
        "story": story,
    }
