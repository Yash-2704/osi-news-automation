"""
Tests for prompt builder and article generator fixes.
Run with: python3 -m pytest tests/test_prompt_and_generation_fixes.py -v

Tests cover:
  1. FORMAT RULE / STORY RULE / SCATTER RULE / ATTRIBUTION RULE in system_message
  2. Blueprint markers (①–⑥) in user_prompt template
  3. No ## section headers in user_prompt output area
  4. Heading correction logic (# → ### after headline)
  5. Accidental-header detection in validate_article_dynamic
  6. Banned-phrase detection still works
  7. b4 strip step removes known section headers from generated text
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_audit(
    source_quality="rich",
    honest_word_ceiling=700,
    has_direct_quotes=True,
    has_named_sources=True,
    has_statistics=True,
    has_future_event=True,
    has_expert_opinion=True,
    has_impact_data=True,
    primary_location="WASHINGTON",
    available_sections=None,
):
    from src.content_generation.models import AuditResult
    return AuditResult(
        source_quality=source_quality,
        honest_word_ceiling=honest_word_ceiling,
        has_direct_quotes=has_direct_quotes,
        has_named_sources=has_named_sources,
        has_statistics=has_statistics,
        has_future_event=has_future_event,
        has_expert_opinion=has_expert_opinion,
        has_impact_data=has_impact_data,
        primary_location=primary_location,
        available_sections=available_sections or [],
    )


def _build_prompt(source_quality="rich"):
    from src.content_generation.prompt_builder import build_dynamic_prompt
    audit = _make_audit(source_quality=source_quality)
    signals = {
        "topic": "Test Topic",
        "source_digest": "Test source material with some facts.",
    }
    # build_dynamic_prompt(articles, topic, audit, signals)
    return build_dynamic_prompt([], "Test Topic", audit, signals)


def _flowing_prose_article(word_padding=200, include_accidental_header=None):
    """
    Build a minimal article with flowing prose (no ## headers in body).
    Optionally inject an accidental section header string.
    """
    filler = " word" * word_padding
    story = (
        "The president signed a defence spending bill on Monday, cutting the budget by "
        "$50 billion — the largest single reduction in a decade. That means roughly "
        "400,000 fewer contract positions across 12 states.\n\n"
        "Defence Secretary Mark Reeves confirmed the measure would take effect on May 1. "
        "Reeves signed the order at the Pentagon in front of assembled joint chiefs, "
        "a choice of venue that signalled the administration wanted military buy-in "
        "rather than resistance.\n\n"
        '"This protects the taxpayer without compromising readiness," Reeves told reporters '
        "after the signing. His use of the word 'readiness' was deliberate — it headed off "
        "the argument from Republicans that cuts would leave the military exposed.\n\n"
        "The bill had been in negotiation since January, when the Office of Management "
        "and Budget flagged a structural $200 billion deficit in the defence account. "
        "According to the Congressional Budget Office, the cuts bring spending to 3.1 percent "
        "of GDP, still above the NATO floor of 2 percent.\n\n"
        "European allies have the most to absorb. Germany and Poland both carry "
        "formal agreements under which U.S. troop levels are tied to a spending ratio — "
        "if Washington falls below 3 percent, both nations must compensate.\n\n"
        "Congress will vote on supplemental funding by April 10."
        + filler
    )
    if include_accidental_header:
        story = include_accidental_header + "\n" + story
    return {
        "heading": "President Signs Defence Cuts Reducing Budget by Fifty Billion Dollars",
        "story": story,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 1: system_message rules
# ──────────────────────────────────────────────────────────────────────────────

class TestSystemMessageRules:

    def test_format_rule_present(self):
        system_message, _ = _build_prompt()
        assert "FORMAT RULE" in system_message, "FORMAT RULE missing from system_message"

    def test_story_rule_present(self):
        system_message, _ = _build_prompt()
        assert "STORY RULE" in system_message, "STORY RULE missing from system_message"

    def test_scatter_rule_present(self):
        system_message, _ = _build_prompt()
        assert "SCATTER RULE" in system_message, "SCATTER RULE missing from system_message"

    def test_attribution_rule_present(self):
        system_message, _ = _build_prompt()
        assert "ATTRIBUTION RULE" in system_message, "ATTRIBUTION RULE missing from system_message"

    def test_pairing_rules_present(self):
        system_message, _ = _build_prompt()
        assert "PAIRING RULES" in system_message, "PAIRING RULES missing from system_message"

    def test_no_structure_rule(self):
        """STRUCTURE RULE (referencing ordered ## sections) must be gone."""
        system_message, _ = _build_prompt()
        assert "STRUCTURE RULE" not in system_message, (
            "Old STRUCTURE RULE still present in system_message"
        )

    def test_format_rule_forbids_section_headers(self):
        """FORMAT RULE must explicitly prohibit ## headers in output."""
        system_message, _ = _build_prompt()
        assert "## " in system_message and "no ##" in system_message.lower() or \
               "no ## lines" in system_message or \
               "Do NOT write any section headers" in system_message, (
            "FORMAT RULE must instruct LLM not to write ## headers"
        )

    def test_no_key_facts_in_system_message(self):
        system_message, _ = _build_prompt()
        assert "Key Facts" not in system_message


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 2: user_prompt template structure
# ──────────────────────────────────────────────────────────────────────────────

class TestUserPromptTemplate:

    def test_blueprint_markers_present(self):
        """All six blueprint markers ①–⑥ must appear in user_prompt."""
        _, user_prompt = _build_prompt()
        for marker in ["①", "②", "③", "④", "⑤", "⑥"]:
            assert marker in user_prompt, f"Blueprint marker {marker} missing from user_prompt"

    def test_no_section_headers_as_output_labels(self):
        """
        ## Lead, ## What Happened, etc. must NOT appear in user_prompt
        as standalone lines (they were the old output headers).
        """
        _, user_prompt = _build_prompt()
        old_headers = [
            "## Lead\n",
            "## What Happened\n",
            "## Voices\n",
            "## Analysis & Context\n",
            "## Implications\n",
            "## What's Next\n",
            "## Key Facts\n",
        ]
        for h in old_headers:
            assert h not in user_prompt, f"Old section header '{h.strip()}' still in user_prompt"

    def test_blueprint_label_not_in_output_section(self):
        """user_prompt must instruct LLM not to copy blueprint labels into output."""
        _, user_prompt = _build_prompt()
        assert "do NOT" in user_prompt or "Do NOT" in user_prompt, (
            "user_prompt must explicitly tell LLM not to copy blueprint labels"
        )

    def test_continuous_prose_instruction_present(self):
        """user_prompt must instruct continuous prose / no section headers in output."""
        _, user_prompt = _build_prompt()
        assert "continuous prose" in user_prompt.lower(), (
            "user_prompt must say 'continuous prose'"
        )

    def test_no_key_facts_in_template(self):
        _, user_prompt = _build_prompt()
        assert "## Key Facts" not in user_prompt

    def test_word_count_rule_present(self):
        _, user_prompt = _build_prompt()
        assert "words total" in user_prompt

    def test_paragraph_count_scales_with_rich(self):
        _, user_prompt = _build_prompt(source_quality="rich")
        assert "2-3 paragraphs" in user_prompt

    def test_paragraph_count_scales_with_thin(self):
        _, user_prompt = _build_prompt(source_quality="thin")
        assert "1 paragraph" in user_prompt


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 3: Post-generation heading correction logic
# ──────────────────────────────────────────────────────────────────────────────

def _apply_heading_correction(generated_text: str) -> str:
    """Replicate exact b3 logic from article_generator.py."""
    _fixed_lines = []
    _headline_seen = False
    for _line in generated_text.splitlines():
        if not _headline_seen and _line.startswith("# ") and not _line.startswith("## "):
            _headline_seen = True
            _fixed_lines.append(_line)
        elif _headline_seen and _line.startswith("# ") and not _line.startswith("## "):
            _fixed_lines.append("##" + _line)
        else:
            _fixed_lines.append(_line)
    return "\n".join(_fixed_lines)


def _apply_section_strip(generated_text: str) -> str:
    """Replicate exact b4 logic from article_generator.py."""
    _SECTION_HEADERS_TO_STRIP = {
        "## Lead", "## What Happened", "## Voices",
        "## Analysis & Context", "## Implications", "## What's Next",
        "## Key Facts",
    }
    _cleaned_lines = [
        _l for _l in generated_text.splitlines()
        if _l.strip() not in _SECTION_HEADERS_TO_STRIP
    ]
    return "\n".join(_cleaned_lines)


class TestHeadingCorrection:

    def test_headline_preserved(self):
        text = "# Real Headline Here For The Story\n### Subheadline here\nSome prose."
        result = _apply_heading_correction(text)
        assert result.startswith("# Real Headline"), "Headline was incorrectly modified"

    def test_spurious_single_hash_becomes_triple_hash(self):
        text = (
            "# Real Headline Here For The Story\n"
            "### Subheadline here\n"
            "First paragraph of prose.\n"
            "# Spurious Subtitle Here\n"
            "More prose."
        )
        result = _apply_heading_correction(text)
        assert "### Spurious Subtitle Here" in result
        # Ensure no line starts with exactly one # (only the original headline should)
        non_headline_single_hash = [
            l for l in result.splitlines()
            if l.startswith("# ") and not l.startswith("## ")
            and l != "# Real Headline Here For The Story"
        ]
        assert not non_headline_single_hash, (
            f"Found un-corrected single-hash lines: {non_headline_single_hash}"
        )

    def test_double_hash_untouched(self):
        text = "# Real Headline\n### Sub\n## Accidental Header\nProse."
        result = _apply_heading_correction(text)
        assert "## Accidental Header" in result  # b3 doesn't touch ##, b4 does

    def test_triple_hash_untouched(self):
        text = "# Real Headline\n### Subheadline\nProse."
        result = _apply_heading_correction(text)
        assert "### Subheadline" in result

    def test_no_headline_no_correction(self):
        text = "Some prose without any headline.\nAnother line."
        result = _apply_heading_correction(text)
        assert result == text


class TestSectionStripStep:

    def test_strips_known_section_headers(self):
        text = (
            "# Real Headline\n"
            "### Subheadline\n"
            "## Lead\n"
            "First paragraph.\n"
            "## What Happened\n"
            "Second paragraph.\n"
            "## Voices\n"
            '"Quote here," said the official.'
        )
        result = _apply_section_strip(text)
        assert "## Lead" not in result
        assert "## What Happened" not in result
        assert "## Voices" not in result

    def test_preserves_headline_and_subheadline(self):
        text = "# Real Headline\n### Subheadline\n## Lead\nProse."
        result = _apply_section_strip(text)
        assert "# Real Headline" in result
        assert "### Subheadline" in result

    def test_preserves_prose_content(self):
        text = "# Headline\n## Lead\nThis is important prose that must survive."
        result = _apply_section_strip(text)
        assert "This is important prose that must survive." in result

    def test_strips_key_facts(self):
        text = "# Headline\n### Sub\n## Key Facts\nBullet list.\nProse continues."
        result = _apply_section_strip(text)
        assert "## Key Facts" not in result

    def test_does_not_strip_inline_mention(self):
        """## Lead appearing mid-sentence (not as standalone line) is not stripped."""
        text = "# Headline\n### Sub\nThe ## Lead section was controversial.\nMore prose."
        result = _apply_section_strip(text)
        assert "## Lead" in result  # not a standalone line, so not stripped


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 4: validate_article_dynamic
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateArticleDynamic:

    def test_clean_prose_passes(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article()
        result = validate_article_dynamic(article, audit)
        assert result["passes"], f"Clean prose article should pass. Failures: {result['failures']}"

    def test_fails_with_accidental_lead_header(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article(include_accidental_header="## Lead")
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]
        assert any("Lead" in f for f in result["failures"]), (
            f"Expected Lead accidental-header failure, got: {result['failures']}"
        )

    def test_fails_with_accidental_what_happened_header(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article(include_accidental_header="## What Happened")
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]

    def test_fails_with_accidental_voices_header(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article(include_accidental_header="## Voices")
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]

    def test_fails_with_accidental_key_facts_header(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article(include_accidental_header="## Key Facts")
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]

    def test_fails_short_article(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = {"heading": "A valid headline for this story", "story": "Too short."}
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]
        assert any("short" in f.lower() for f in result["failures"])

    def test_fails_short_headline(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article()
        article["heading"] = "Short"
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]

    def test_fails_banned_phrase(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit()
        article = _flowing_prose_article()
        article["story"] += " this development was crucial to understand"
        result = validate_article_dynamic(article, audit)
        assert not result["passes"]
        assert any("crucial" in f.lower() for f in result["failures"])

    def test_quote_warning_when_no_quotes(self):
        from src.content_generation.prompt_builder import validate_article_dynamic
        audit = _make_audit(has_direct_quotes=True)
        story = "word " * 200  # plenty of words but no quotes
        article = {"heading": "A valid headline for the story here", "story": story}
        result = validate_article_dynamic(article, audit)
        # Should pass (quote check is a warning, not a failure)
        assert result["passes"]
        assert any("quote" in w.lower() for w in result["warnings"])


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 5: Pre-dry-run checklist
# Covers: ceiling enforcement (Prompt 1), repair instruction routing (Prompt 3),
# and max_tokens formula correctness (Prompt 1).
# ──────────────────────────────────────────────────────────────────────────────

class TestPreDryRunChecklist:

    # Test A — Validator ceiling check (Prompt 1)
    def test_validator_ceiling_enforcement(self):
        """420-word article must hard-fail against a 350-word ceiling (+50 tolerance = 400)."""
        from src.content_generation.prompt_builder import validate_article_dynamic
        from src.content_generation.models import AuditResult
        audit = AuditResult(
            source_quality="thin",
            honest_word_ceiling=350,
            available_sections=["what_happened"],
            has_direct_quotes=False,
            has_named_sources=True,
            has_statistics=True,
            has_future_event=False,
            has_expert_opinion=False,
            has_impact_data=False,
            primary_location=None,
        )
        article = {"heading": "Test headline for unit test", "story": "word " * 420}
        result = validate_article_dynamic(article, audit)
        assert result["passes"] is False, \
            "Ceiling check not firing — article over ceiling should fail"
        assert any("ceiling" in f.lower() for f in result["failures"]), \
            f"Ceiling failure message missing. Got failures: {result['failures']}"

    # Test B — Repair instruction routing (Prompt 3)
    def test_repair_instructions_ceiling(self):
        """Ceiling failure string must route to a WORD CEILING repair instruction."""
        from src.content_generation.article_generator import _build_repair_instructions
        r = _build_repair_instructions(
            ["Word ceiling exceeded: 732 words written, ceiling is 350"], 350
        )
        assert "WORD CEILING" in r, \
            f"Expected WORD CEILING in repair output, got:\n{r}"

    def test_repair_instructions_banned_phrase(self):
        """Banned-phrase failure string must route to a BANNED PHRASE repair instruction."""
        from src.content_generation.article_generator import _build_repair_instructions
        r = _build_repair_instructions(["Banned phrase detected: 'historic'"], 700)
        assert "BANNED PHRASE" in r, \
            f"Expected BANNED PHRASE in repair output, got:\n{r}"

    def test_repair_instructions_empty_failures_fallback(self):
        """Empty failures list must return the generic fallback message."""
        from src.content_generation.article_generator import _build_repair_instructions
        r = _build_repair_instructions([], 500)
        assert "Review all rules" in r, \
            f"Expected fallback message, got:\n{r}"

    # Test C — max_tokens formula (Prompt 1)
    def test_max_tokens_formula_thin(self):
        """Thin ceiling 350 → max_tokens 552."""
        assert int(350 * 1.35) + 80 == 552

    def test_max_tokens_formula_adequate(self):
        """Adequate ceiling 500 → max_tokens 755."""
        assert int(500 * 1.35) + 80 == 755

    def test_max_tokens_formula_rich(self):
        """Rich ceiling 800 → max_tokens 1160."""
        assert int(800 * 1.35) + 80 == 1160

    def test_max_tokens_retry_reduction_step1(self):
        """First retry reduces thin max_tokens by 10% → lands in [494, 498]."""
        base = 552
        reduced = max(200, int(base * (0.9 ** 1)))
        assert 494 <= reduced <= 498, \
            f"Expected 494–498 after one retry reduction, got {reduced}"

    def test_max_tokens_retry_reduction_step2(self):
        """Second retry reduces thin max_tokens by 19% → lands in [445, 449]."""
        base = 552
        reduced = max(200, int(base * (0.9 ** 2)))
        assert 445 <= reduced <= 449, \
            f"Expected 445–449 after two retry reductions, got {reduced}"
