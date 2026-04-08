# Clustering & Prompting Architecture Overview

This document outlines how the OSI News Automation system processes unstructured scraped articles into coherent, clustered trends, and generates high-quality articles using a dynamic, two-stage prompting architecture.

## 1. Trend Detection & Clustering (`trend_analyzer.py`)

The trend detection phase groups newly scraped articles based on semantic similarity.

### Pre-processing & Embeddings
- The system concatenates the `heading` and the first 300 characters of the `story` for each scraped article. This snippet is used because comparing headlines alone is often too sparse to obtain meaningful similarity matches.
- Using `sentence-transformers` (`all-MiniLM-L6-v2`), it calculates embedding vectors for these snippets. If the model fails to load, the system falls back to a sequence-matching (`difflib`) baseline.

### Clustering Approach
- **Affinity/Similarity:** Evaluated using a Cosine Similarity matrix across the valid snippets.
- **Algorithm:** Uses `AgglomerativeClustering` (Hierarchical clustering) from `scikit-learn` with `average` linkage.
- **Threshold Limit:** Clustering relies on `distance_threshold = 1.0 - similarity_threshold` (default similarity threshold is `0.3`). This avoids forcing categorically unrelated articles into a single, diluted cluster.
- **Filtering:** Clusters below `min_cluster_size` (default `2`) are discarded.

### Topic Naming
Topic labels are generated using a 3-tier cascade:
1. **LLM Generation:** Up to 8 headlines from the cluster are sent to Groq (`llama-3.3-70b-versatile` by default), asking for a descriptive, sub-15-word title.
2. **Centroid Match:** If the LLM call fails, the system calculates the geometric centroid of the cluster's embeddings and grabs the "closest" matching headline, truncating it cleanly at natural boundaries (like `-`, `|`, or `,`).
3. **Fallback:** Purely truncates the first headline to 10 words.

---

## 2. Dynamic, Two-Stage Content Generation (`article_generator.py` & `prompt_builder.py`)

The pipeline was redesigned to utilize a "Two-Stage Audit" model. This specifically addresses problems with content hallucination and poor article quality ("thin" source material).

### Stage 1: The Audit Phase (`audit_source_material()`)
Before writing an article, a maximum of 12 internal source articles (up to 1,500 characters each) are digested and passed to the LLM (using the `instructor` library for strict schema output). The model produces an `AuditResult` specifying:
- **`source_quality`**: Evaluates the digest as either `rich`, `adequate`, or `thin`.
- **`honest_word_ceiling`**: Calculates the maximum number of words realistically sustainable by the source facts (e.g., `rich`: 700-900 words, `thin`: 180-350 words).
- **Extracted Attributes**: Detects `has_direct_quotes`, `has_named_sources`, `has_future_event`, `has_expert_opinion`, etc.
- **Hard Block:** If the source quality returns as `thin` with fewer than 2 available sections, generation is gracefully scrubbed and falls back to a simpler summary.

### Stage 2: Synthesis & Structure (`build_dynamic_prompt()`)
The actual article generation prompt is assembled based on the extracted `AuditResult` and specific rules:
- **Variable Constraints:** Narrative depth heavily scales to the `source_quality`. For example, `thin` constraints explicitly forbid speculation and require concise 1-paragraph limits per section.
- **Signal Passing:** The user prompt is divided into 7 non-negotiable sections:
  1. `## Lead`
  2. `## Key Facts` (Only confirmed bullets)
  3. `## What Happened`
  4. `## Voices` (Uses direct quotes if audit confirms them, else paraphrased attribution)
  5. `## Analysis & Context`
  6. `## Implications`
  7. `## What's Next`
- **Rigid Tone Checks:** Enforces AP Style and completely restricts vague summarizing.

---

## 3. Article Content Problems & Mitigations (`validate_article_dynamic`)

To address recurring content problems (e.g., poor pacing, source bias leaks, hallucinated conclusions), a strict post-generation validator evaluates the LLM's output. Any "Hard Failures" trigger a complete regeneration retry (up to 3 tries total).

### Hard Failures (Rejects Article)
- **Minimum Word Bounds:** The system evaluates `honest_word_ceiling`. If the generated response falls below `150` words despite this, it's discarded.
- **Section Dropping:** Every one of the 7 sections *must* exist within the generated markdown structure exactly as requested.
- **Banned Phrases:** Overworked filler phrases strictly block the article. This includes items like *"landmark"*, *"unprecedented"*, *"regional and global consequences"*, *"it remains to be seen"*, etc. Removing these forces the LLM to lean exclusively on extracted facts.

### Warnings (Logged, Accepted Article)
- **Topic Drift:** Automatically compares paragraphs against topic keywords. If a paragraph doesn't share enough relation to the original topic phrase, a warning is logged.
- **Outlet Leaks:** Ensures brand names like `Reuters`, `CNN`, `BBC` are filtered out of the actual article body. 
- **Meta-Commentary / Scaffolding:** Warns if the LLM leaked instructional padding (like *"...not yet clear from available reporting"* or *"sources do not contain"*). 

### How to Address Remaining Content Problems:
If you are still seeing problems with generated articles:
1. **Model Limits:** `llama-3.3-70b-versatile` may be struggling to adhere strictly to the schema. You might want to evaluate the prompt instructions in `build_dynamic_prompt()`.
2. **Review Validation Rules:** If additional "filler" vocabulary keeps showing up in generations, add it directly to the `BANNED_PHRASES` list in `src/content_generation/prompt_builder.py`.
3. **Tweak Distance Threshold:** If articles within clusters don't actually share an exact topic, lower the `similarity_threshold` in `detect_trends` to prevent unrelated topics grouping.
