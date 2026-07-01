"""Filesystem caching utilities for API calls and heavy data processing."""

import time
import logging
from typing import Callable, Any
from pathlib import Path
import joblib
from src.config import CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cached_or_fetch(key: str, fetch_fn: Callable[[], Any], ttl_hours: int = 24) -> Any:
    """
    Fetches data from a local joblib cache if valid, otherwise executes fetch_fn.
    
    Args:
        key: Unique identifier string for the cached artifact.
        fetch_fn: Function to execute if cache misses.
        ttl_hours: Time-to-live for the cache in hours.
        
    Returns:
        The cached or freshly fetched data object.
    """
    cache_path = CACHE_DIR / f"{key}.pkl"
    ttl_seconds = ttl_hours * 3600

    if cache_path.exists():
        file_age = time.time() - cache_path.stat().st_mtime
        if file_age < ttl_seconds:
            logger.info(f"Cache HIT for key: {key}")
            return joblib.load(cache_path)
        else:
            logger.info(f"Cache STALE for key: {key} (Age: {file_age:.1f}s). Re-fetching.")
    else:
        logger.info(f"Cache MISS for key: {key}. Fetching.")

    data = fetch_fn()
    joblib.dump(data, cache_path)
    logger.info(f"Cache SAVED for key: {key}")
    return data