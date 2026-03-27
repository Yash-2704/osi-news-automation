"""
OSI News Automation System - Trend Analyzer
============================================
Analyzes scraped articles to identify trending topics using NLP.
Uses sentence embeddings and clustering to group similar articles.
"""

# Import logger before try block so it's available in the except clause.
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_EMBEDDINGS = True
# Capture the actual error so logs show the real cause
# rather than a generic "not installed" message.
except ImportError as e:
    logger.warning(
        f"sentence-transformers unavailable ({e}), "
        f"falling back to keyword similarity"
    )
    _HAS_EMBEDDINGS = False
    SentenceTransformer = None

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ===========================================
# MODEL LOADING (Lazy initialization)
# ===========================================

_model = None


def get_model():
    """
    Get or initialize the sentence transformer model.
    Uses lazy loading to avoid startup delay.
    Returns None if sentence-transformers is not installed.
    """
    global _model

    if not _HAS_EMBEDDINGS:
        return None

    if _model is None:
        logger.info("Loading sentence transformer model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")

    return _model


# ===========================================
# KEYWORD-BASED SIMILARITY FALLBACK
# ===========================================

def _keyword_similarity(text_a: str, text_b: str) -> float:
    """Fallback text similarity using difflib sequence matching."""
    if not text_a or not text_b:
        return 0.0
        
    import difflib
    # Compare raw strings (case-insensitive)
    return difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


# ===========================================
# STOP WORDS
# ===========================================

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'you', 'your',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'new',
    'says', 'said', 'say', 'after', 'before', 'over', 'under', 'again',
    'further', 'into', 'through', 'during', 'above', 'below', 'between',
    'about', 'against', 'news', 'report', 'reports', 'latest', 'update',
}


# ===========================================
# KEYWORD EXTRACTION
# ===========================================

def extract_keywords(articles: List[Dict], top_n: int = 10) -> List[str]:
    """
    Extract common keywords from a cluster of articles.
    
    Args:
        articles: List of article dictionaries.
        top_n: Number of top keywords to return.
        
    Returns:
        List of most common keywords.
    """
    # Combine headlines and story previews
    all_text = ' '.join([
        a.get('heading', '') + ' ' + a.get('story', '')[:300] 
        for a in articles
    ])
    
    # Extract words (alphanumeric, 3+ chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # Filter stop words
    words = [w for w in words if w not in STOP_WORDS]
    
    # Count frequencies
    word_counts = Counter(words)
    
    return [word for word, count in word_counts.most_common(top_n)]


def extract_topic_name(articles: List[Dict]) -> str:
    """
    Extract a representative topic name from a cluster of articles.
    
    Picks the headline whose embedding is closest to the cluster
    centroid, then returns its first 7 words.  Falls back to
    frequency-based word selection when embeddings are unavailable.
    
    Args:
        articles: List of article dictionaries in the cluster.
        
    Returns:
        A topic name string.
    """
    # Step A — Centroid-nearest headline (requires embeddings + >1 article)
    try:
        model = get_model()
        if model is not None and len(articles) > 1:
            headlines = [a.get('heading', '') for a in articles]
            non_empty = [(i, h) for i, h in enumerate(headlines) if h.strip()]
            if non_empty:
                texts = [h for _, h in non_empty]
                embeddings = model.encode(texts, show_progress_bar=False)
                centroid = np.mean(embeddings, axis=0)
                # Closest headline to centroid by L2 distance
                dists = [np.linalg.norm(emb - centroid) for emb in embeddings]
                best_idx = int(np.argmin(dists))
                return ' '.join(texts[best_idx].split()[:7])
    except Exception:
        logger.debug("extract_topic_name: embedding path failed, using fallback")

    # Step B — Fallback: top-4 most common significant words
    all_headlines = ' '.join([a.get('heading', '') for a in articles])
    words = re.findall(r'\b[A-Za-z]{4,}\b', all_headlines)
    words = [w for w in words if w.lower() not in STOP_WORDS]

    if words:
        word_counts = Counter(words)
        top_words = [word for word, _ in word_counts.most_common(4)]
        return ' '.join(top_words)

    # Step C — Final fallback: first 6 words of first headline
    if articles:
        first_headline = articles[0].get('heading', 'Unknown Topic')
        return ' '.join(first_headline.split()[:6])

    return "General News"


# ===========================================
# TREND DETECTION
# ===========================================

def detect_trends(
    articles: List[Dict],
    top_n: int = 5,
    min_cluster_size: int = 2,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    Detect trending topics from scraped articles using NLP clustering.
    
    Process:
    1. Generate embeddings for all headlines
    2. Calculate similarity matrix
    3. Cluster similar articles using Agglomerative Clustering
    4. Rank clusters by size
    5. Extract topic names and keywords
    
    Args:
        articles: List of article dictionaries (must have 'heading' key).
        top_n: Number of top trends to return.
        min_cluster_size: Minimum articles to form a trend.
        similarity_threshold: Minimum similarity to consider related.
        
    Returns:
        List of trend dictionaries with:
        - topic: Descriptive topic name
        - article_count: Number of articles in cluster
        - keywords: List of related keywords
        - articles: List of articles in the trend
        - avg_similarity: Average similarity within cluster
        - first_seen: Earliest article timestamp
        - sources: List of source names
        
    Example:
        >>> trends = detect_trends(scraped_articles, top_n=3)
        >>> for trend in trends:
        ...     print(f"{trend['topic']}: {trend['article_count']} articles")
    """
    if not articles:
        logger.warning("No articles provided for trend detection")
        return []
    
    if len(articles) < min_cluster_size:
        logger.warning(f"Not enough articles for trend detection (need {min_cluster_size})")
        return []
    
    try:
        logger.info(f"Analyzing {len(articles)} articles for trends...")
        
        # Extract headlines
        headlines = [article.get('heading', '') for article in articles]
        
        # Filter out empty headlines
        valid_indices = [i for i, h in enumerate(headlines) if h.strip()]
        if len(valid_indices) < min_cluster_size:
            logger.warning("Not enough valid headlines for trend detection")
            return []
        
        valid_headlines = [headlines[i] for i in valid_indices]
        valid_articles = [articles[i] for i in valid_indices]
        
        # Generate similarity matrix (embeddings or keyword fallback)
        model = get_model()
        if model is not None:
            logger.debug("Generating embeddings...")
            embeddings = model.encode(valid_headlines, show_progress_bar=False)
            similarity_matrix = cosine_similarity(embeddings)
        else:
            logger.info("Using keyword-based similarity (sentence-transformers not available)")
            n = len(valid_headlines)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    sim = _keyword_similarity(valid_headlines[i], valid_headlines[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
                similarity_matrix[i][i] = 1.0
        
        # Convert to distance matrix for clustering
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
        
        # Perform clustering
        if _HAS_EMBEDDINGS:
            # Use distance threshold so only genuinely similar articles
            #  cluster together; avoids forcing unrelated articles into one group.
            computed_distance_threshold = 1.0 - similarity_threshold
            logger.debug(f"Clustering with distance_threshold={computed_distance_threshold:.2f}...")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=computed_distance_threshold,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
        else:
            # Simple greedy clustering fallback
            logger.debug("Using greedy clustering fallback...")
            labels = [-1] * len(valid_articles)
            cluster_id = 0
            for i in range(len(valid_articles)):
                if labels[i] != -1:
                    continue
                labels[i] = cluster_id
                for j in range(i + 1, len(valid_articles)):
                    if labels[j] == -1 and similarity_matrix[i][j] >= similarity_threshold:
                        labels[j] = cluster_id
                cluster_id += 1
        
        # Group articles by cluster
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Rank clusters by size
        cluster_sizes = [(label, len(indices)) for label, indices in clusters.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Build trend objects for top clusters
        trends = []
        
        for label, size in cluster_sizes:
            if size < min_cluster_size:
                continue
            
            if len(trends) >= top_n:
                break
            
            article_indices = clusters[label]
            cluster_articles = [valid_articles[i] for i in article_indices]
            
            # Calculate average similarity within cluster
            if len(article_indices) > 1:
                similarities = [
                    similarity_matrix[i][j]
                    for i in article_indices
                    for j in article_indices
                    if i != j
                ]
                avg_similarity = float(np.mean(similarities)) if similarities else 0.0
            else:
                avg_similarity = 1.0
            
            # Extract topic and keywords
            topic = extract_topic_name(cluster_articles)
            keywords = extract_keywords(cluster_articles)
            
            # Get sources
            sources = list(set([
                a.get('source_name', 'Unknown') 
                for a in cluster_articles
            ]))
            
            # Get time range
            timestamps = []
            for a in cluster_articles:
                if 'scraped_at' in a:
                    timestamps.append(a['scraped_at'])
                elif 'publish_date' in a:
                    timestamps.append(a['publish_date'])
            
            first_seen = min(timestamps) if timestamps else datetime.utcnow().isoformat()
            
            trend = {
                "topic": topic,
                "article_count": size,
                "keywords": keywords,
                "articles": cluster_articles,
                "article_ids": [a.get('_id') for a in cluster_articles if '_id' in a],
                "avg_similarity": round(avg_similarity, 3),
                "first_seen": first_seen,
                "sources": sources,
                "detected_at": datetime.utcnow().isoformat()
            }
            
            trends.append(trend)
            logger.info(f"📈 Trend: '{topic}' ({size} articles, {len(sources)} sources)")
        
        logger.info(f"Detected {len(trends)} trending topics")
        return trends
        
    except Exception as e:
        logger.error(f"Trend detection failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


def find_related_articles(
    target_article: Dict,
    articles: List[Dict],
    top_n: int = 5,
    min_similarity: float = 0.5
) -> List[Tuple[Dict, float]]:
    """
    Find articles related to a target article.
    
    Args:
        target_article: Article to find related articles for.
        articles: Pool of articles to search.
        top_n: Maximum number of related articles to return.
        min_similarity: Minimum similarity threshold.
        
    Returns:
        List of (article, similarity_score) tuples.
    """
    if not target_article or not articles:
        return []
    
    try:
        model = get_model()
        
        target_headline = target_article.get('heading', '')
        if not target_headline:
            return []
        
        # Get headlines
        headlines = [target_headline] + [a.get('heading', '') for a in articles]
        
        # Generate embeddings
        embeddings = model.encode(headlines, show_progress_bar=False)
        
        # Calculate similarities with target
        target_embedding = embeddings[0].reshape(1, -1)
        other_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(target_embedding, other_embeddings)[0]
        
        # Get top related articles
        related = []
        for idx, sim in enumerate(similarities):
            if sim >= min_similarity:
                related.append((articles[idx], float(sim)))
        
        # Sort by similarity
        related.sort(key=lambda x: x[1], reverse=True)
        
        return related[:top_n]
        
    except Exception as e:
        logger.error(f"Error finding related articles: {e}")
        return []


# ===========================================
# TESTING
# ===========================================

def test_trend_detection():
    """Test trend detection with sample articles."""
    print("\n" + "="*60)
    print("🧪 Trend Detection Test")
    print("="*60)
    
    # Create test articles with obvious clustering
    test_articles = [
        # Cluster 1: Russia/Putin (3 articles)
        {"heading": "Putin warns of military action in Eastern Europe", "story": "Russian president issues warning..."},
        {"heading": "Russia threatens Ukraine with new sanctions", "story": "Moscow officials announced..."},
        {"heading": "Putin issues stern warning to NATO allies", "story": "The Kremlin stated today..."},
        
        # Cluster 2: India Economy (3 articles)
        {"heading": "India GDP growth accelerates to 7.5%", "story": "Economy shows strong performance..."},
        {"heading": "Indian economy shows remarkable strength", "story": "Markets rally on positive data..."},
        {"heading": "India's economic growth exceeds expectations", "story": "Analysts surprised by results..."},
        
        # Cluster 3: Climate (2 articles)
        {"heading": "Climate summit reaches historic agreement", "story": "World leaders agree on emissions..."},
        {"heading": "Global climate talks produce breakthrough", "story": "Environmental groups celebrate..."},
        
        # Noise (unrelated articles)
        {"heading": "New iPhone features leaked online", "story": "Tech rumors suggest..."},
        {"heading": "Local sports team wins championship", "story": "Fans celebrate victory..."},
    ]
    
    print(f"\n📰 Testing with {len(test_articles)} sample articles...")
    print("-" * 40)
    
    trends = detect_trends(test_articles, top_n=3, min_cluster_size=2)
    
    if trends:
        print(f"\n✅ Detected {len(trends)} trends:\n")
        
        for i, trend in enumerate(trends, 1):
            print(f"{i}. {trend['topic']}")
            print(f"   Articles: {trend['article_count']}")
            print(f"   Similarity: {trend['avg_similarity']:.2%}")
            print(f"   Keywords: {', '.join(trend['keywords'][:5])}")
            print()
    else:
        print("❌ No trends detected")
    
    print("="*60 + "\n")
    
    return trends


if __name__ == "__main__":
    test_trend_detection()
