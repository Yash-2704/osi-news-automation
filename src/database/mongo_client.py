"""
OSI News Automation System - MongoDB Client
============================================
Comprehensive MongoDB client for managing articles, trends, and scraping sessions.
Supports both local MongoDB and MongoDB Atlas.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError, ServerSelectionTimeoutError
from datetime import datetime, timedelta
from loguru import logger
from bson import ObjectId
import os
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MongoDBClient:
    """
    MongoDB client for OSI News Automation System.
    
    Handles all database operations including:
    - Article storage and retrieval
    - Trend tracking
    - Scraping session management
    - Duplicate detection using sentence embeddings
    - Upload status tracking
    """
    
    def __init__(self, uri: str = None, database_name: str = None):
        """
        Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection URI. If None, reads from environment.
            database_name: Database name. If None, reads from environment.
        """
        # Load from environment if not provided
        # MONGO_URI = cloud Atlas connection string (set on Render/production)
        # MONGODB_LOCAL_URI = localhost fallback for local development
        self.uri = uri or os.getenv("MONGO_URI") or os.getenv("MONGODB_LOCAL_URI", "mongodb://localhost:27017/")
        self.database_name = database_name or os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DATABASE", "osi_news_automation")
        
        # Collection names from environment
        self.articles_collection_name = os.getenv("MONGODB_COLLECTION_ARTICLES", "articles")
        self.trends_collection_name = os.getenv("MONGODB_COLLECTION_TRENDS", "trends")
        self.sessions_collection_name = os.getenv("MONGODB_COLLECTION_SESSIONS", "scraping_sessions")
        
        # Client and database references
        self.client: Optional[MongoClient] = None
        self.db = None
        self.articles = None
        self.trends = None
        self.sessions = None
        
        # Sentence transformer for duplicate detection (lazy loaded)
        self._embedding_model = None
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Connection state
        self._connected = False
        
        logger.info(f"MongoDBClient initialized for database: {self.database_name}")
    
    @property
    def embedding_model(self):
        """Lazy load the sentence transformer model (if available)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence transformer model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            # Log the actual ImportError message so the real cause is
            # visible in logs rather than the misleading "not installed".
            except ImportError as e:
                logger.warning(
                    f"sentence-transformers unavailable ({e}), "
                    f"embedding-based dedup disabled"
                )
        return self._embedding_model
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to MongoDB at {self.uri[:30]}...")
            
            # Create client with connection pooling
            # Enable TLS only for remote (Atlas) connections, not localhost
            is_local = "localhost" in self.uri or "127.0.0.1" in self.uri
            connect_kwargs = dict(
                serverSelectionTimeoutMS=20000,  # 20 second timeout
                connectTimeoutMS=20000,
                maxPoolSize=50,
                retryWrites=True,
            )
            if not is_local:
                import certifi
                connect_kwargs["tls"] = True
                connect_kwargs["tlsCAFile"] = certifi.where()

            self.client = MongoClient(self.uri, **connect_kwargs)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database and collections
            self.db = self.client[self.database_name]
            self.articles = self.db[self.articles_collection_name]
            self.trends = self.db[self.trends_collection_name]
            self.sessions = self.db[self.sessions_collection_name]
            
            # Create indexes for performance
            self._create_indexes()
            
            self._connected = True
            logger.success(f"Connected to MongoDB database: {self.database_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self._connected = False
            return False
    
    def _create_indexes(self) -> None:
        """Create database indexes for optimal performance."""
        try:
            # Articles indexes
            self.articles.create_index([("session_id", ASCENDING)])
            self.articles.create_index([("scraped_at", DESCENDING)])
            self.articles.create_index([("upload_status", ASCENDING)])
            self.articles.create_index([("source_url", ASCENDING)], unique=True, sparse=True)
            # Retry queue index
            self.articles.create_index([("upload_status", ASCENDING), ("upload_last_retry", ASCENDING)])
            # Prompt debug index — sparse so older articles without prompt_debug
            # are not included, avoiding null-key index bloat
            self.articles.create_index(
                [("prompt_debug.captured_at", DESCENDING)],
                sparse=True,
                name="prompt_debug_captured_at"
            )
            
            # Trends indexes — topic is unique (match existing DB index)
            self.trends.create_index([("topic", ASCENDING)], unique=True)
            self.trends.create_index([("last_seen", DESCENDING)])
            
            # Sessions indexes
            self.sessions.create_index([("session_id", ASCENDING)], unique=True)
            self.sessions.create_index([("started_at", DESCENDING)])
            
            logger.debug("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes (may already exist): {e}")
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")
    
    def _coerce_datetime(self, value) -> datetime:
        """
        Coerce a value to a datetime object.

        MongoDB's $jsonSchema 'bsonType: date' requires an actual datetime,
        not an ISO string. Translators and generators sometimes store
        dates as isoformat() strings — this corrects that.
        """
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # Handle both '2026-02-20T10:26:17.765387' and '2026-02-20T10:26:17Z'
                return datetime.fromisoformat(value.replace('Z', '+00:00').replace('+00:00', ''))
            except ValueError:
                pass
        return datetime.utcnow()

    def _ensure_connected(self) -> bool:
        """Ensure database is connected, attempt reconnection if needed."""
        if not self._connected or self.client is None:
            logger.warning("Database not connected, attempting to reconnect...")
            return self.connect()
        
        try:
            # Ping to verify connection is alive
            self.client.admin.command('ping')
            return True
        except Exception:
            logger.warning("Connection lost, attempting to reconnect...")
            return self.connect()
    
    # ===========================================
    # ARTICLE OPERATIONS
    # ===========================================
    
    def save_article(self, article_dict: Dict[str, Any]) -> Optional[str]:
        """
        Save an article to the database.
        
        Args:
            article_dict: Article data dictionary.
            
        Returns:
            str: Inserted article ID, or None on failure.
        """
        if not self._ensure_connected():
            return None
        
        try:
            # Add metadata with retry tracking fields
            article = {
                **article_dict,
                # Ensure scraped_at is always a BSON date, never an ISO string
                "scraped_at": self._coerce_datetime(article_dict.get("scraped_at", datetime.utcnow())),
                "upload_status": article_dict.get("upload_status", "pending"),
                # Retry tracking fields
                "upload_retry_count": article_dict.get("upload_retry_count", 0),
                "upload_last_retry": article_dict.get("upload_last_retry", None),
                "upload_failure_reason": article_dict.get("upload_failure_reason", None)
            }

            # MongoDB uses the `language` field as a text-index language specifier.
            # Values like 'hi' (Hindi) and 'ar' (Arabic) are not supported and cause
            # error 17262. Copy to `content_language` and remove from the DB document.
            # Note: `article` is already a shallow copy ({**article_dict, ...}) so
            # this does not mutate the caller's original dictionary.
            if "language" in article:
                article["content_language"] = article["language"]
                del article["language"]

            # Only include hocalwire_feed_id if it has a real value (schema requires string, not null)
            if article_dict.get("hocalwire_feed_id"):
                article["hocalwire_feed_id"] = article_dict["hocalwire_feed_id"]
            
            # Generate embedding for duplicate detection if story exists
            if "story" in article and article["story"]:
                emb = self._get_embedding(article["story"])
                if emb is not None:
                    article["embedding"] = emb
            
            result = self.articles.insert_one(article)
            article_id = str(result.inserted_id)
            
            logger.debug(f"Saved article: {article.get('heading', 'Unknown')[:50]}...")
            return article_id
            
        except DuplicateKeyError:
            logger.warning(f"Duplicate article detected: {article_dict.get('source_url', 'Unknown')}")
            return None
        except Exception as e:
            logger.error(f"Error saving article: {e}")
            return None
    
    def get_article_by_id(self, article_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        Retrieve an article by its ID.
        
        Args:
            article_id: Article ObjectId or string ID.
            
        Returns:
            dict: Article data, or None if not found.
        """
        if not self._ensure_connected():
            return None
        
        try:
            if isinstance(article_id, str):
                article_id = ObjectId(article_id)
            
            article = self.articles.find_one({"_id": article_id})
            return article
            
        except Exception as e:
            logger.error(f"Error retrieving article {article_id}: {e}")
            return None
    
    def get_articles_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all articles from a specific scraping session.
        
        Args:
            session_id: The session identifier.
            
        Returns:
            list: List of article dictionaries.
        """
        if not self._ensure_connected():
            return []
        
        try:
            articles = list(self.articles.find(
                {"session_id": session_id}
            ).sort("scraped_at", DESCENDING))
            
            logger.debug(f"Retrieved {len(articles)} articles for session {session_id}")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles for session {session_id}: {e}")
            return []
    
    def get_recent_articles(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get articles scraped within the specified number of hours.
        
        Args:
            hours: Number of hours to look back.
            
        Returns:
            list: List of recent articles.
        """
        if not self._ensure_connected():
            return []
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            articles = list(self.articles.find(
                {"scraped_at": {"$gte": cutoff_time}}
            ).sort("scraped_at", DESCENDING))
            
            logger.debug(f"Retrieved {len(articles)} articles from last {hours} hours")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving recent articles: {e}")
            return []
    
    def bulk_insert_articles(self, articles_list: List[Dict[str, Any]]) -> int:
        """
        Bulk insert multiple articles.
        
        Args:
            articles_list: List of article dictionaries.
            
        Returns:
            int: Number of successfully inserted articles.
        """
        if not self._ensure_connected():
            return 0
        
        if not articles_list:
            return 0
        
        try:
            # Prepare articles with metadata and embeddings
            prepared_articles = []
            for article in articles_list:
                prepared = {
                    **article,
                    # Ensure scraped_at is always a BSON date, never an ISO string
                    "scraped_at": self._coerce_datetime(article.get("scraped_at", datetime.utcnow())),
                    "upload_status": article.get("upload_status", "pending"),
                    # Retry tracking fields
                    "upload_retry_count": article.get("upload_retry_count", 0),
                    "upload_last_retry": article.get("upload_last_retry", None),
                    "upload_failure_reason": article.get("upload_failure_reason", None)
                }
                # Rename `language` to avoid MongoDB text-index language override error 17262
                if "language" in prepared:
                    prepared["content_language"] = prepared.pop("language")
                
                # Generate embedding if story exists
                if "story" in prepared and prepared["story"]:
                    prepared["embedding"] = self._get_embedding(prepared["story"])
                
                prepared_articles.append(prepared)
            
            result = self.articles.insert_many(prepared_articles, ordered=False)
            inserted_count = len(result.inserted_ids)
            
            logger.info(f"Bulk inserted {inserted_count} articles")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error bulk inserting articles: {e}")
            return 0
    
    def update_upload_status(
        self, 
        article_id: Union[str, ObjectId], 
        status: str,
        hocalwire_feed_id: str = None,
        failure_reason: str = None,
        increment_retry: bool = False
    ) -> bool:
        """
        Update the upload status of an article.
        
        Args:
            article_id: Article ID.
            status: New status ("pending", "uploaded", "failed", "retry_exhausted").
            hocalwire_feed_id: Optional Hocalwire feed ID if uploaded.
            failure_reason: Optional failure reason if failed.
            increment_retry: Whether to increment retry count.
            
        Returns:
            bool: True if update successful.
        """
        if not self._ensure_connected():
            return False
        
        try:
            if isinstance(article_id, str):
                article_id = ObjectId(article_id)
            
            update_data = {
                "upload_status": status,
            }
            
            # Handle successful upload
            if status == "uploaded":
                update_data["uploaded_at"] = datetime.utcnow()
                # Clear retry tracking on success
                update_data["upload_retry_count"] = 0
                update_data["upload_last_retry"] = None
                update_data["upload_failure_reason"] = None
            
            # Handle failed upload
            elif status == "failed":
                update_data["upload_last_retry"] = datetime.utcnow()
                if failure_reason:
                    update_data["upload_failure_reason"] = failure_reason
            
            # Add Hocalwire feed ID if provided
            if hocalwire_feed_id:
                update_data["hocalwire_feed_id"] = hocalwire_feed_id
            
            # Build update operation
            update_operation = {"$set": update_data}
            
            # Increment retry count if requested
            if increment_retry:
                update_operation["$inc"] = {"upload_retry_count": 1}
            
            result = self.articles.update_one(
                {"_id": article_id},
                update_operation
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating upload status for {article_id}: {e}")
            return False
    
    def get_failed_uploads(
        self,
        max_retries: int = 10,
        min_retry_interval_minutes: int = 15,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get articles with failed uploads that are eligible for retry.
        
        Args:
            max_retries: Maximum retry attempts before giving up.
            min_retry_interval_minutes: Minimum minutes since last retry.
            limit: Maximum number of articles to return.
            
        Returns:
            list: List of articles eligible for retry.
        """
        if not self._ensure_connected():
            return []
        
        try:
            # Calculate cutoff time for retry attempts
            retry_cutoff = datetime.utcnow() - timedelta(minutes=min_retry_interval_minutes)
            
            # Query for failed uploads eligible for retry
            query = {
                "upload_status": "failed",
                "upload_retry_count": {"$lt": max_retries},
                "$or": [
                    {"upload_last_retry": None},
                    {"upload_last_retry": {"$lt": retry_cutoff}}
                ]
            }
            
            articles = list(self.articles.find(query).limit(limit))
            
            logger.debug(f"Found {len(articles)} failed uploads eligible for retry")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving failed uploads: {e}")
            return []
    

    # ===========================================
    # DUPLICATE DETECTION
    # ===========================================
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text, with caching.
        Returns None if sentence-transformers is not installed.
        """
        model = self.embedding_model
        if model is None:
            return None
        
        # Use hash of first 500 chars as cache key
        cache_key = str(hash(text[:500]))
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = model.encode(text).tolist()
        
        # Limit cache size
        if len(self._embedding_cache) > 1000:
            # Remove oldest half of cache
            keys_to_remove = list(self._embedding_cache.keys())[:500]
            for key in keys_to_remove:
                del self._embedding_cache[key]
        
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def check_duplicate(self, article_text: str, similarity_threshold: float = 0.85, exclude_session_id: str = None) -> bool:
        """
        Check if an article is a duplicate of existing articles.
        
        Compares with articles from the last 48 hours using sentence embeddings.
        If sentence-transformers is not available, falls back to difflib.SequenceMatcher.
        
        Args:
            article_text: Text content of the article.
            similarity_threshold: Minimum similarity to consider duplicate (0.0-1.0).
            exclude_session_id: If provided, exclude articles from this session
                                to prevent same-run false positives.
            
        Returns:
            bool: True if duplicate found, False otherwise.
        """
        if not self._ensure_connected():
            return False
        
        if not article_text or len(article_text.strip()) < 50:
            return False
        
        try:
            # Get embedding for new article
            new_embedding = self._get_embedding(article_text)
            
            # Get recent articles (last 48 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            
            # Base query: generated articles from the last 48 hours
            base_query = {
                "scraped_at": {"$gte": cutoff_time},
                "pipeline_stage": "generated",
            }
            # Exclude articles from the current pipeline session to avoid
            # two related-but-different articles flagging each other.
            if exclude_session_id:
                base_query["session_id"] = {"$ne": exclude_session_id}
            
            # If embeddings are available, use them
            if new_embedding is not None:
                emb_query = {**base_query, "embedding": {"$exists": True}}
                recent_articles = self.articles.find(
                    emb_query,
                    {"embedding": 1, "heading": 1}
                )
                
                for article in recent_articles:
                    if "embedding" in article and article["embedding"]:
                        similarity = self._cosine_similarity(new_embedding, article["embedding"])
                        
                        if similarity >= similarity_threshold:
                            logger.debug(
                                f"Duplicate detected (embedding similarity: {similarity:.2f}): "
                                f"{article.get('heading', 'Unknown')[:50]}..."
                            )
                            return True
            else:
                # Lightweight Fallback: Use difflib + keyword similarity
                import difflib
                import re
                
                _STOP = {'the','a','an','and','or','but','in','on','at','to','for',
                         'of','with','by','from','as','is','was','are','were','be',
                         'has','have','had','its','it','this','that','not','no',
                         'says','said','say','after','over','into','news','update',
                         'government','president','minister','officials','people',
                         'country','world','report','reports','reported','new',
                         'state','states','national','international','according'}
                
                def _heading_jaccard(h1: str, h2: str) -> float:
                    """Word-set overlap (Jaccard) — robust to word reordering by LLM."""
                    w1 = set(re.findall(r'[a-z]{3,}', h1.lower())) - _STOP
                    w2 = set(re.findall(r'[a-z]{3,}', h2.lower())) - _STOP
                    if not w1 or not w2:
                        return 0.0
                    return len(w1 & w2) / len(w1 | w2)
                
                # IMPORTANT: Only compare against previously GENERATED articles,
                # NOT raw scraped articles (which are saved in the same pipeline run).
                fallback_query = {**base_query, "heading": {"$exists": True}}
                recent_articles = self.articles.find(
                    fallback_query,
                    {"story": 1, "heading": 1}
                )
                
                # Content similarity threshold (strict — character-level match)
                content_threshold = min(0.9, similarity_threshold + 0.1)
                # Heading keyword overlap threshold
                # 0.65 = requires ~2/3 keyword overlap to flag as duplicate
                heading_threshold = 0.65
                
                # Extract heading from input for heading-vs-heading comparison
                # Input may be "heading text story text..." — take first line
                input_heading = article_text.split('\n')[0][:150].strip()
                
                for article in recent_articles:
                    existing_heading = article.get("heading", "")
                    
                    # 1) Heading keyword overlap (catches same-topic rewrites by LLM)
                    if existing_heading and input_heading:
                        heading_sim = _heading_jaccard(input_heading, existing_heading)
                        if heading_sim >= heading_threshold:
                            logger.info(
                                f"Duplicate detected (heading keyword overlap: {heading_sim:.2f}): "
                                f"{existing_heading[:60]}..."
                            )
                            return True
                    
                    # 2) Content-to-content comparison (catches exact reposts)
                    existing_text = article.get("story", "")
                    if existing_text:
                        similarity = difflib.SequenceMatcher(None, article_text[:500], existing_text[:500]).ratio()
                        if similarity >= content_threshold:
                            logger.info(
                                f"Duplicate detected (content similarity: {similarity:.2f}): "
                                f"{existing_heading[:60]}..."
                            )
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False
    
    # ===========================================
    # TREND OPERATIONS
    # ===========================================
    
    def save_trend(self, trend_dict: Dict[str, Any]) -> Optional[str]:
        """
        Save or update a trend topic.
        
        Args:
            trend_dict: Trend data with topic, keywords, etc.
            
        Returns:
            str: Trend ID, or None on failure.
        """
        if not self._ensure_connected():
            return None
        
        try:
            topic = trend_dict.get("topic", "")
            
            # Try to update existing trend
            result = self.trends.update_one(
                {"topic": topic},
                {
                    "$set": {
                        "last_seen": datetime.utcnow(),
                        "keywords": trend_dict.get("keywords", [])
                    },
                    "$inc": {"article_count": 1},
                    "$addToSet": {
                        "related_articles": {"$each": trend_dict.get("related_articles", [])}
                    },
                    "$setOnInsert": {
                        "first_seen": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            if result.upserted_id:
                return str(result.upserted_id)
            
            # Return existing trend ID
            existing = self.trends.find_one({"topic": topic})
            return str(existing["_id"]) if existing else None
            
        except Exception as e:
            logger.error(f"Error saving trend: {e}")
            return None
    
    def get_active_trends(self, hours: int = 24, min_articles: int = 3) -> List[Dict[str, Any]]:
        """
        Get currently active trending topics.
        
        Args:
            hours: Look back period in hours.
            min_articles: Minimum articles to be considered trending.
            
        Returns:
            list: List of active trends sorted by article count.
        """
        if not self._ensure_connected():
            return []
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            trends = list(self.trends.find(
                {
                    "last_seen": {"$gte": cutoff_time},
                    "article_count": {"$gte": min_articles}
                }
            ).sort("article_count", DESCENDING))
            
            logger.debug(f"Retrieved {len(trends)} active trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error retrieving active trends: {e}")
            return []
    
    # ===========================================
    # SCRAPING SESSION OPERATIONS
    # ===========================================
    
    def save_scraping_session(self, session_dict: Dict[str, Any]) -> Optional[str]:
        """
        Save a scraping session record.
        
        Args:
            session_dict: Session data including session_id, sources, etc.
            
        Returns:
            str: Session ID, or None on failure.
        """
        if not self._ensure_connected():
            return None
        
        try:
            session = {
                **session_dict,
                "started_at": session_dict.get("started_at", datetime.utcnow()),
                "status": session_dict.get("status", "running")
            }
            
            result = self.sessions.insert_one(session)
            session_id = str(result.inserted_id)
            
            logger.info(f"Created scraping session: {session.get('session_id', session_id)}")
            return session_id
            
        except DuplicateKeyError:
            logger.warning(f"Session already exists: {session_dict.get('session_id')}")
            return session_dict.get('session_id')
        except Exception as e:
            logger.error(f"Error saving scraping session: {e}")
            return None
    
    def update_session_status(
        self, 
        session_id: str, 
        status: str, 
        articles_count: int = None,
        error_message: str = None
    ) -> bool:
        """
        Update the status of a scraping session.
        
        Args:
            session_id: The session identifier.
            status: New status ("running", "completed", "failed").
            articles_count: Optional count of scraped articles.
            error_message: Optional error message if failed.
            
        Returns:
            bool: True if update successful.
        """
        if not self._ensure_connected():
            return False
        
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status in ["completed", "failed"]:
                update_data["ended_at"] = datetime.utcnow()
            
            if articles_count is not None:
                update_data["articles_count"] = articles_count
            
            if error_message:
                update_data["error_message"] = error_message
            
            result = self.sessions.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a scraping session by ID.
        
        Args:
            session_id: The session identifier.
            
        Returns:
            dict: Session data, or None if not found.
        """
        if not self._ensure_connected():
            return None
        
        try:
            return self.sessions.find_one({"session_id": session_id})
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    # ===========================================
    # STATISTICS & UTILITIES
    # ===========================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            dict: Statistics including counts and recent activity.
        """
        if not self._ensure_connected():
            return {}
        
        try:
            now = datetime.utcnow()
            last_24h = now - timedelta(hours=24)
            
            stats = {
                "total_articles": self.articles.count_documents({}),
                "articles_last_24h": self.articles.count_documents(
                    {"scraped_at": {"$gte": last_24h}}
                ),
                "pending_uploads": self.articles.count_documents(
                    {"upload_status": "pending"}
                ),
                "uploaded_articles": self.articles.count_documents(
                    {"upload_status": "uploaded"}
                ),
                "active_trends": self.trends.count_documents(
                    {"last_seen": {"$gte": last_24h}}
                ),
                "total_sessions": self.sessions.count_documents({}),
                "database_name": self.database_name,
                "connected": self._connected
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def delete_old_articles(self, days: int = 30) -> int:
        """
        Delete articles older than specified days.
        
        Args:
            days: Delete articles older than this many days.
            
        Returns:
            int: Number of deleted articles.
        """
        if not self._ensure_connected():
            return 0
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            result = self.articles.delete_many(
                {"scraped_at": {"$lt": cutoff_time}}
            )
            
            deleted_count = result.deleted_count
            logger.info(f"Deleted {deleted_count} articles older than {days} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting old articles: {e}")
            return 0


# ===========================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ===========================================

_default_client: Optional[MongoDBClient] = None


def get_client() -> MongoDBClient:
    """Get or create the default MongoDB client."""
    global _default_client
    
    if _default_client is None:
        _default_client = MongoDBClient()
        _default_client.connect()
    
    return _default_client


def close_client() -> None:
    """Close the default MongoDB client."""
    global _default_client
    
    if _default_client:
        _default_client.disconnect()
        _default_client = None
