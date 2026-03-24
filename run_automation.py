#!/usr/bin/env python3
"""
OSI News Automation System - Main Pipeline Orchestrator
========================================================
Master script that orchestrates the complete news automation pipeline:
1. Scrape articles from multiple sources
2. Detect trending topics
3. Generate comprehensive articles
4. Create AI-generated images
5. Translate to multiple languages
6. Upload to Hocalwire CMS
7. Generate social media posts
8. Run on schedule or on-demand

Usage:
    python run_automation.py --mode once        # Run once
    python run_automation.py --mode scheduled   # Run every 3 hours
    python run_automation.py --mode dry-run     # Test without uploads
"""

import sys
import os
import subprocess
import argparse
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# -------------------------------------------------------
# Ensure all critical dependencies are installed BEFORE
# importing any src.* modules. This fixes GitHub Actions
# runner environments where some packages may be missing.
# -------------------------------------------------------
def ensure_dependencies():
    """Install missing packages before any src imports."""
    required = {
        "langdetect": "langdetect==1.0.9",
        "feedparser": "feedparser==6.0.10",
        "cloudinary": "cloudinary==1.36.0",
    }
    missing = []
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[bootstrap] Installing missing packages: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )

ensure_dependencies()
# -------------------------------------------------------

from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline components
from src.scrapers.batch_scraper import scrape_news_batch
from src.trend_detection.trend_analyzer import detect_trends
from src.content_generation.article_generator import generate_article
from src.image_generation.image_creator import initialize_sd_pipeline, generate_article_image
from src.translation.translator import translate_article
from src.database.mongo_client import MongoDBClient
from src.api_integrations.hocalwire_uploader import upload_batch_to_hocalwire, generate_session_id
from src.api_integrations.social_media_poster import generate_social_posts


# ===========================================
# LOGGING CONFIGURATION
# ===========================================

def setup_logging():
    """Configure logging for the pipeline."""
    # Remove default logger
    logger.remove()
    
    # Console logging (INFO level)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )
    
    # File logging (DEBUG level)
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / "automation_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info("Logging configured successfully")


# ===========================================
# PIPELINE EXECUTION
# ===========================================

def run_pipeline(dry_run: bool = False) -> dict:
    """
    Execute the complete news automation pipeline.
    
    Args:
        dry_run: If True, skip actual uploads and external API calls.
        
    Returns:
        Dictionary with pipeline statistics and results.
    """
    logger.info("=" * 80)
    logger.info("🚀 OSI NEWS AUTOMATION PIPELINE STARTING")
    logger.info("=" * 80)
    
    pipeline_start = datetime.now()
    session_id = generate_session_id()
    
    # Pipeline statistics
    stats = {
        'session_id': session_id,
        'started_at': pipeline_start.isoformat(),
        'articles_scraped': 0,
        'trends_detected': 0,
        'articles_generated': 0,
        'articles_skipped_duplicate': 0,
        'images_created': 0,
        'translations_created': 0,
        'uploads_successful': 0,
        'social_posts_generated': 0,
        'errors': []
    }
    
    try:
        # Initialize database
        logger.info("\n🔌 Connecting to database...")
        db = MongoDBClient()
        if not db.connect():
            raise Exception("Failed to connect to MongoDB")
        logger.info("✅ Database connected")
        
        # Initialize image generation (if enabled)
        if os.getenv('ENABLE_IMAGE_GENERATION', 'false').lower() == 'true' and not dry_run:
            logger.info("\n🎨 Initializing Stable Diffusion...")
            initialize_sd_pipeline()
        
        # ==========================================
        # STEP 1: SCRAPE ARTICLES
        # ==========================================
        logger.info("\n" + "=" * 80)
        logger.info("📰 STEP 1: SCRAPING NEWS ARTICLES")
        logger.info("=" * 80)
        
        max_articles = int(os.getenv('MAX_ARTICLES_PER_RUN', 50))
        logger.info(f"Target: {max_articles} articles")
        
        articles = scrape_news_batch(max_articles=max_articles)
        
        if not articles:
            logger.error("❌ No articles scraped. Aborting pipeline.")
            stats['errors'].append("No articles scraped")
            return stats
        
        stats['articles_scraped'] = len(articles)
        logger.info(f"✅ Scraped {len(articles)} articles")
        
        # Save raw articles to database
        logger.info("💾 Saving articles to database...")
        for article in articles:
            article['session_id'] = session_id
            article['pipeline_stage'] = 'scraped'
            db.save_article(article)
        
        # ==========================================
        # STEP 2: DETECT TRENDS
        # ==========================================
        logger.info("\n" + "=" * 80)
        logger.info("🔍 STEP 2: DETECTING TRENDING TOPICS")
        logger.info("=" * 80)
        
        top_n_trends = int(os.getenv('TOP_TRENDS_COUNT', 5))
        min_cluster_size = int(os.getenv('MIN_CLUSTER_SIZE', 1))
        similarity_threshold = float(os.getenv('DUPLICATE_SIMILARITY_THRESHOLD', 0.6))

        # Use the actual NLP clustering from trend_analyzer to group similar stories into single trends
        trends = detect_trends(
            articles,
            top_n=top_n_trends,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold
        )
        
        stats['trends_detected'] = len(trends)
        logger.info(f"✅ Detected {len(trends)} distinct trends from {len(articles)} articles (duplicates merged):")
        for i, trend in enumerate(trends, 1):
            logger.info(f"   {i}. {trend['topic']} ({trend['article_count']} sources)")
        
        # Save trends to database
        for trend in trends:
            trend['session_id'] = session_id
            db.save_trend(trend)
        
        # ==========================================
        # STEP 3: GENERATE ARTICLES
        # ==========================================
        logger.info("\n" + "=" * 80)
        logger.info("✍️ STEP 3: GENERATING COMPREHENSIVE ARTICLES")
        logger.info("=" * 80)
        
        generated_articles = []
        image_urls = {}
        
        for i, trend in enumerate(trends):
            logger.info(f"\n📝 Processing trend {i+1}/{len(trends)}: {trend['topic']}")
            
            # ── Single-source guard ──
            # Never generate from fewer than 2 source articles.
            # A single-source article cannot be honestly synthesized
            # to 800+ words — the model will fabricate to fill the gap.
            source_count = len(trend.get('articles', []))
            if source_count < 2:
                logger.warning(
                    f"⏭️  Skipping '{trend['topic']}' — "
                    f"only {source_count} source(s). "
                    f"Minimum 2 required for honest synthesis."
                )
                stats['errors'].append(
                    f"Skipped (single source): {trend['topic']}"
                )
                continue
            
            try:
                # Generate article
                target_words = int(os.getenv('ARTICLE_MIN_WORDS', 800))
                article = generate_article(trend, target_words=target_words)
                
                if not article:
                    logger.warning(f"⚠️ Failed to generate article for: {trend['topic']}")
                    stats['errors'].append(f"Article generation failed: {trend['topic']}")
                    continue
                
                # Attach session metadata before duplicate check
                article['session_id'] = session_id
                article['pipeline_stage'] = 'generated'
                article['trend_index'] = i
                
                # ── Post-generation duplicate check ──────────────────
                # Compare the LLM-generated text against previously
                # generated LLM text in the DB (like vs like).
                generated_text = article.get('heading', '') + ' ' + article.get('story', '')
                dup_threshold = float(os.getenv('DUPLICATE_SIMILARITY_THRESHOLD', 0.75))
                if db.check_duplicate(generated_text, similarity_threshold=dup_threshold, exclude_session_id=session_id):
                    logger.warning(
                        f"⏭️ Skipping trend '{trend['topic']}' — "
                        f"similar article already exists in database (post-generation check)"
                    )
                    stats['articles_skipped_duplicate'] += 1
                    continue
                
                generated_articles.append(article)
                
                logger.info(f"✅ Generated article: {article['heading'][:60]}...")
                logger.info(f"   Words: {article.get('word_count', 0)}")
                
                # Save to database
                article_id = db.save_article(article)
                article['_id'] = article_id
                
            except Exception as e:
                logger.error(f"❌ Error generating article for trend '{trend['topic']}': {e}")
                stats['errors'].append(f"Article generation error: {str(e)}")
                continue
        
        stats['articles_generated'] = len(generated_articles)
        logger.info(f"\n✅ Generated {len(generated_articles)} comprehensive articles")
        
        # ==========================================
        # STEP 4: GENERATE IMAGES
        # ==========================================
        if os.getenv('ENABLE_IMAGE_GENERATION', 'false').lower() == 'true' and not dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("🎨 STEP 4: GENERATING AI IMAGES")
            logger.info("=" * 80)
            
            # Import Cloudinary uploader
            from src.image_generation.cloudinary_uploader import upload_image_to_cloudinary
            
            for i, article in enumerate(generated_articles):
                try:
                    logger.info(f"\n🖼️ Generating image {i+1}/{len(generated_articles)}...")
                    image_path = generate_article_image(article)
                    
                    if image_path:
                        # Upload to Cloudinary and get public URL
                        logger.info("📤 Uploading image to Cloudinary...")
                        public_url = upload_image_to_cloudinary(
                            image_path,
                            folder="osi-news",
                            public_id=f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        )
                        
                        if public_url:
                            image_urls[i] = public_url
                            article['image_url'] = public_url
                            article['image_path'] = image_path  # Keep local path for reference
                            stats['images_created'] += 1
                            logger.info(f"✅ Image uploaded: {public_url}")
                        else:
                            logger.warning(f"⚠️ Cloudinary upload failed, using fallback")
                            image_urls[i] = "https://www.hocalwire.com/images/logo.png"
                    else:
                        logger.warning(f"⚠️ Image generation failed for article {i+1}")
                        
                except Exception as e:
                    logger.error(f"❌ Error generating/uploading image: {e}")
                    stats['errors'].append(f"Image generation error: {str(e)}")
        else:
            logger.info("\n⏭️ STEP 4: Image generation skipped (disabled or dry-run)")
        
        # ==========================================
        # STEP 5: TRANSLATE ARTICLES
        # ==========================================
        if os.getenv('TRANSLATION_ENABLED', 'false').lower() == 'true' and not dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("🌐 STEP 5: TRANSLATING ARTICLES")
            logger.info("=" * 80)
            
            for i, article in enumerate(generated_articles):
                try:
                    logger.info(f"\n🗣️ Translating article {i+1}/{len(generated_articles)}...")
                    translations = translate_article(article)
                    
                    for lang, translated in translations.items():
                        translated['session_id'] = session_id
                        translated['pipeline_stage'] = 'translated'
                        translated['original_article_id'] = article.get('_id')
                        db.save_article(translated)
                        stats['translations_created'] += 1
                    
                    if translations:
                        logger.info(f"✅ Translated to {len(translations)} languages: {', '.join(translations.keys())}")
                    
                except Exception as e:
                    logger.error(f"❌ Error translating article: {e}")
                    stats['errors'].append(f"Translation error: {str(e)}")
        else:
            logger.info("\n⏭️ STEP 5: Translation skipped (disabled or dry-run)")
        
        # ==========================================
        # STEP 6: UPLOAD TO HOCALWIRE
        # ==========================================
        if not dry_run and os.getenv('ENABLE_HOCALWIRE_UPLOAD', 'true').lower() == 'true':
            logger.info("\n" + "=" * 80)
            logger.info("📤 STEP 6: UPLOADING TO HOCALWIRE")
            logger.info("=" * 80)
            
            try:
                upload_stats = upload_batch_to_hocalwire(
                    generated_articles,
                    image_urls=image_urls,
                    max_retries=3
                )
                
                stats['uploads_successful'] = upload_stats.get('successful', 0)
                logger.info(f"✅ Upload complete: {upload_stats['successful']}/{upload_stats['total']} successful")
                
                if upload_stats.get('failed', 0) > 0:
                    stats['errors'].append(f"{upload_stats['failed']} uploads failed")
                    
            except Exception as e:
                logger.error(f"❌ Error uploading to Hocalwire: {e}")
                stats['errors'].append(f"Hocalwire upload error: {str(e)}")
        else:
            logger.info("\n⏭️ STEP 6: Hocalwire upload skipped (dry-run or disabled)")
        
        # ==========================================
        # STEP 7: GENERATE SOCIAL MEDIA POSTS
        # ==========================================
        logger.info("\n" + "=" * 80)
        logger.info("📱 STEP 7: GENERATING SOCIAL MEDIA POSTS")
        logger.info("=" * 80)
        
        all_social_posts = []
        
        for i, article in enumerate(generated_articles):
            try:
                # Construct article URL
                feed_id = article.get('hocalwire_feed_id', f'local_{i}')
                article_url = f"https://democracynewslive.com/article/{feed_id}"
                image_url = image_urls.get(i, "")
                
                # Generate posts
                posts = generate_social_posts(article, article_url, image_url)
                
                all_social_posts.append({
                    'article_id': str(article.get('_id', '')),
                    'article_title': article['heading'],
                    'article_url': article_url,
                    'posts': posts,
                    'generated_at': datetime.now().isoformat()
                })
                
                stats['social_posts_generated'] += 1
                logger.info(f"✅ Generated social posts for: {article['heading'][:50]}...")
                
            except Exception as e:
                logger.error(f"❌ Error generating social posts: {e}")
                stats['errors'].append(f"Social post generation error: {str(e)}")
        
        # Save social posts to JSON
        if all_social_posts:
            social_posts_dir = Path("output/json")
            social_posts_dir.mkdir(parents=True, exist_ok=True)
            social_posts_file = social_posts_dir / f"social_posts_{session_id}.json"
            
            with open(social_posts_file, 'w', encoding='utf-8') as f:
                json.dump(all_social_posts, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Social posts saved: {social_posts_file}")
        
        # ==========================================
        # PIPELINE COMPLETE
        # ==========================================
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        stats['completed_at'] = datetime.now().isoformat()
        stats['duration_seconds'] = pipeline_duration
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📊 Session ID: {session_id}")
        logger.info(f"⏱️  Duration: {pipeline_duration:.1f} seconds ({pipeline_duration/60:.1f} minutes)")
        logger.info(f"📰 Articles Scraped: {stats['articles_scraped']}")
        logger.info(f"🔍 Trends Detected: {stats['trends_detected']}")
        logger.info(f"✍️  Articles Generated: {stats['articles_generated']}")
        logger.info(f"⏭️  Duplicates Skipped: {stats['articles_skipped_duplicate']}")
        logger.info(f"🎨 Images Created: {stats['images_created']}")
        logger.info(f"🌐 Translations Created: {stats['translations_created']}")
        logger.info(f"📤 Uploads Successful: {stats['uploads_successful']}")
        logger.info(f"📱 Social Posts Generated: {stats['social_posts_generated']}")        
        if stats['errors']:
            logger.warning(f"⚠️  Errors Encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"   - {error}")
        
        logger.info("=" * 80)
        
        # Save pipeline stats
        stats_dir = Path("output/json")
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / f"pipeline_stats_{session_id}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"📊 Pipeline stats saved: {stats_file}")
        
        return stats
        
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        stats['completed_at'] = datetime.now().isoformat()
        stats['duration_seconds'] = (datetime.now() - pipeline_start).total_seconds()
        stats['errors'].append(f"Pipeline failure: {str(e)}")
        stats['status'] = 'failed'
        
        raise


def scheduled_pipeline():
    """Wrapper for scheduled execution."""
    logger.info("\n⏰ Scheduled run triggered at {}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    try:
        run_pipeline(dry_run=False)
    except Exception as e:
        logger.error(f"Scheduled pipeline failed: {e}")
        # In production, send alert email/Slack notification here


# ===========================================
# MAIN ENTRY POINT
# ===========================================

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='OSI News Automation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_automation.py --mode once        # Run pipeline once
  python run_automation.py --mode scheduled   # Run every 3 hours
  python run_automation.py --mode dry-run     # Test without uploads
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['once', 'scheduled', 'dry-run'],
        default='once',
        help='Execution mode: once (single run), scheduled (every N hours), or dry-run (test mode)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("OSI News Automation System v1.0")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.mode == 'once':
            logger.info("Running pipeline once...")
            run_pipeline(dry_run=False)
            
        elif args.mode == 'dry-run':
            logger.info("Running pipeline in DRY-RUN mode (no uploads)...")
            run_pipeline(dry_run=True)
            
        elif args.mode == 'scheduled':
            logger.info("Starting scheduled pipeline...")
            
            try:
                from apscheduler.schedulers.blocking import BlockingScheduler
            except ImportError:
                logger.error("APScheduler not installed. Run: pip install apscheduler")
                sys.exit(1)
            
            # Import retry service
            from src.api_integrations.retry_failed_uploads import run_retry_queue
            
            scheduler = BlockingScheduler()
            
            # Get intervals from environment
            interval_hours = int(os.getenv('SCRAPING_INTERVAL_HOURS', 3))
            retry_interval_minutes = int(os.getenv('RETRY_INTERVAL_MINUTES', 30))
            retry_enabled = os.getenv('RETRY_FAILED_UPLOADS_ENABLED', 'true').lower() == 'true'
            
            # Schedule main pipeline job
            scheduler.add_job(
                scheduled_pipeline,
                'interval',
                hours=interval_hours,
                next_run_time=datetime.now()  # Run immediately on start
            )
            
            # Schedule retry queue job (if enabled)
            if retry_enabled:
                def retry_job():
                    """Wrapper for retry queue execution."""
                    logger.info("\\n⏰ Retry queue triggered at {}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    try:
                        run_retry_queue(dry_run=False)
                    except Exception as e:
                        logger.error(f"Retry queue failed: {e}")
                
                scheduler.add_job(
                    retry_job,
                    'interval',
                    minutes=retry_interval_minutes,
                    next_run_time=datetime.now() + timedelta(minutes=5)  # Start 5 min after launch
                )
                
                logger.info(f"⏰ Retry queue configured: Running every {retry_interval_minutes} minutes")
            else:
                logger.info("⏭️ Retry queue disabled (RETRY_FAILED_UPLOADS_ENABLED=false)")
            
            logger.info(f"⏰ Main pipeline configured: Running every {interval_hours} hours")
            logger.info("Press Ctrl+C to stop")
            
            try:
                scheduler.start()
            except (KeyboardInterrupt, SystemExit):
                logger.info("\n🛑 Scheduler stopped by user")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
