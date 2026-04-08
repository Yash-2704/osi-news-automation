"""
OSI News Automation – Shared Models
====================================
Pydantic models shared between article_generator.py and prompt_builder.py.
Defined in a separate module to avoid circular imports.
"""

from typing import List, Optional

from pydantic import BaseModel


class AuditResult(BaseModel):
    available_sections: List[str]
    # List of section keys the source material can honestly support.
    # Valid values are exactly: "what_happened", "key_facts",
    # "who_is_affected", "background_context", "reactions",
    # "expert_analysis", "looking_ahead"
    # Only include a section if sources genuinely support it.

    has_direct_quotes: bool
    # True only if the source material contains text inside
    # quotation marks attributed to a named person

    has_named_sources: bool
    # True if any named person or institution appears in sources

    has_statistics: bool
    # True if any numeric figure appears in sources

    has_future_event: bool
    # True if any upcoming deadline, date, or decision is mentioned

    has_expert_opinion: bool
    # True if any analytical/interpretive statement by a named expert exists

    has_impact_data: bool
    # True if any consequence on another country, market, or population is stated

    primary_location: Optional[str]
    # Most specific location where events are occurring, or None

    source_quality: str
    # Must be exactly one of: "rich", "adequate", "thin"
    # rich = multiple sources with quotes, stats, named people
    # adequate = some facts and named sources, limited quotes
    # thin = minimal facts, no quotes, sparse detail

    honest_word_ceiling: int
    # Realistic maximum word count this material can support
    # without fabrication. rich=700-900, adequate=400-600, thin=200-400
