import time
import hashlib
import json
import os
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from config.settings import settings


class CacheManager:
    """Intelligent caching system for agent responses"""

    def __init__(self):
        self.cache_dir = "data/cache"
        self.ttl_seconds = settings.CACHE_TTL_SECONDS
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        cache_key = self._get_cache_key(query)
        cache_file = self._get_cache_file(cache_key)

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl_seconds):
                os.remove(cache_file)
                return None

            return cached_data["result"]

        except Exception:
            return None

    def set(self, query: str, result: Dict[str, Any]):
        """Cache a result"""
        cache_key = self._get_cache_key(query)
        cache_file = self._get_cache_file(cache_key)

        cached_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result
        }

        try:
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
        except Exception:
            pass  # Fail silently if caching fails

    def clear_expired(self):
        """Clear expired cache entries"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        cached_data = json.load(f)

                    cached_time = datetime.fromisoformat(cached_data["timestamp"])
                    if datetime.now() - cached_time > timedelta(seconds=self.ttl_seconds):
                        os.remove(filepath)
                except Exception:
                    continue

    def clear_all(self):
        """Clear all cache entries"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, filename))