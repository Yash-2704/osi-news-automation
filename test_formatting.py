"""
Test the updated formatting function for semantic HTML output.

Covers: ## headers, key facts bullet list, **bold** inline, body paragraphs.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api_integrations.hocalwire_uploader import format_article_for_cms

# ── Sample article covering all four formatting scenarios ──
sample_story = """HAWAII, March 22 – A powerful flash flooding event has struck Oahu.

**Key facts:**
- Evacuation orders affect 5,500 people.
- Flooding is the worst in 20 years.
- At least 200 people rescued.
- 10 hospitalized with hypothermia.

## The Event

The flash flooding began **early Saturday morning**, with heavy rains pounding
the north shore of Oahu.

## Damage and Casualties

At least 200 people have been rescued. Fortunately, no deaths reported.

## Context and Analysis

The flooding highlights the importance of **infrastructure resilience**.

## What Happens Next

FEMA is scheduled to conduct a damage assessment on March 25."""

# ── Run formatter ──
formatted = format_article_for_cms(sample_story)

print("=" * 80)
print("FORMATTED HTML OUTPUT")
print("=" * 80)
print()
print(formatted)
print()
print("=" * 80)

# ── Assertions ──
assert "<h2" in formatted,       "FAIL: no <h2> tags — header conversion missing"
assert "<ul" in formatted,       "FAIL: no <ul> — bullet list conversion missing"
assert "<li" in formatted,       "FAIL: no <li> — list items missing"
assert "<strong>" in formatted,  "FAIL: no <strong> — bold conversion missing"
assert "- " not in formatted,    "FAIL: raw dash-space still present"
assert "**" not in formatted,    "FAIL: raw bold markers still present"

print("✅ All assertions passed")

# ── Edge case: empty input ──
assert format_article_for_cms("") == "", "FAIL: empty input should return empty string"
assert format_article_for_cms(None) == "", "FAIL: None input should return empty string"
print("✅ Edge case assertions passed (empty/None input)")
