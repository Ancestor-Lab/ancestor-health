"""
Phase 3: Deterministic Source Caching Service
Implements URL-normalized cache keys with fail-closed behavior.

Key Features:
- SHA-256 of normalized URL (strips tracking params)
- TTL-based expiration (24-48h standard, <4h news)
- Fail-closed: Redis unavailable -> defaults to live fetch
- Malformed cache entries treated as fetch failures
"""

import hashlib
import json
import logging
import time
from urllib.parse import urlparse, urlunparse, parse_qs
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Tracking parameters to strip for URL normalization
TRACKING_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'fbclid', 'gclid', 'msclkid', '_ga', 'mc_eid', 'mc_cid'
}

class CacheService:
    """
    Deterministic source caching with URL normalization and fail-closed behavior.
    """

    def __init__(self, host='localhost', port=6379, password=None, db=0):
        """
        Initialize cache service with Redis backend.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password (optional)
            db: Redis database number
        """
        self.enabled = False
        self.redis = None

        try:
            import redis
            self.redis = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,
                socket_timeout=2.0,  # Fail fast if Redis is down
                socket_connect_timeout=2.0
            )

            # Test connection
            self.redis.ping()
            self.enabled = True
            logger.info(f"✅ Cache Service ONLINE: Redis at {host}:{port}")

        except ImportError:
            logger.warning("⚠️ Cache Service DISABLED: Redis module not available")
        except Exception as e:
            logger.warning(f"⚠️ Cache Service DISABLED: Redis connection failed - {str(e)}")
            logger.info("🔄 Defaulting to live fetch only (fail-closed behavior)")

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL to ensure stable cache keys.

        Normalization steps:
        1. Convert scheme and netloc to lowercase
        2. Remove fragment
        3. Remove tracking parameters
        4. Sort remaining query parameters
        5. Normalize path

        Args:
            url: Original URL

        Returns:
            Normalized URL for cache key generation
        """
        try:
            parsed = urlparse(url)

            # Normalize scheme and netloc to lowercase
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()

            # Normalize path (remove trailing slash except for root)
            path = parsed.path
            if path != '/' and path.endswith('/'):
                path = path.rstrip('/')

            # Filter and sort query parameters
            filtered_params = []
            if parsed.query:
                query_dict = parse_qs(parsed.query, keep_blank_values=False)
                for key, values in query_dict.items():
                    if key.lower() not in TRACKING_PARAMS:
                        # Sort values for deterministic ordering
                        for value in sorted(values):
                            filtered_params.append(f"{key}={value}")

            # Sort parameters for deterministic ordering
            filtered_params.sort()
            clean_query = '&'.join(filtered_params)

            # Reconstruct normalized URL
            normalized = urlunparse((
                scheme,
                netloc,
                path,
                parsed.params,  # Keep params as-is
                clean_query,
                ''  # Remove fragment
            ))

            return normalized

        except Exception as e:
            logger.warning(f"URL normalization failed for {url}: {str(e)}")
            return url  # Return original URL if normalization fails

    def _generate_cache_key(self, url: str) -> str:
        """
        Generate deterministic cache key from normalized URL.

        Args:
            url: Original URL

        Returns:
            Cache key (prefixed, versioned, hashed)
        """
        normalized_url = self._normalize_url(url)
        url_hash = hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
        return f"ancestor:source:v3:{url_hash}"

    def _determine_ttl(self, url: str, default_ttl: int = 86400) -> int:
        """
        Determine TTL based on URL characteristics.

        Args:
            url: Source URL
            default_ttl: Default TTL in seconds (24 hours)

        Returns:
            TTL in seconds
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # News/fast-changing sources get shorter TTL
            news_domains = {
                'news.', 'www.reuters.', 'www.ap.org', 'www.bbc.',
                'cnn.com', 'www.nytimes.', 'www.washingtonpost.',
                'www.wsj.', 'www.bloomberg.'
            }

            for news_domain in news_domains:
                if news_domain in domain:
                    return 14400  # 4 hours for news sources

            # Standard sources: 24-48 hours
            return default_ttl  # 24 hours default

        except Exception:
            return default_ttl

    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached content for URL.

        Args:
            url: Source URL to look up

        Returns:
            Cached data if available and valid, None otherwise
        """
        if not self.enabled:
            return None

        try:
            cache_key = self._generate_cache_key(url)
            cached_data = self.redis.get(cache_key)

            if not cached_data:
                logger.debug(f"⚪ Cache MISS: {url}")
                return None

            # Parse cached data
            try:
                data = json.loads(cached_data)
            except json.JSONDecodeError as e:
                logger.warning(f"🔥 Malformed cache entry for {url}: {str(e)}")
                # Treat malformed cache as fetch failure (fail-closed)
                self.redis.delete(cache_key)
                return None

            # Validate cache entry structure
            required_fields = ['extracted_text', 'fetch_timestamp', 'availability_signal']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"🔥 Invalid cache entry for {url}: missing {field}")
                    self.redis.delete(cache_key)
                    return None

            # Check for malformed content
            extracted_text = data.get('extracted_text', '')
            if data['availability_signal'] > 0 and len(extracted_text) < 50:
                logger.warning(f"🔥 Suspicious cache entry for {url}: content too short")
                self.redis.delete(cache_key)
                return None

            logger.info(f"🟢 Cache HIT: {url}")
            return data

        except Exception as e:
            logger.error(f"Cache read error for {url}: {str(e)}")
            return None

    def set(self, url: str, fetch_result: Dict[str, Any]) -> bool:
        """
        Store fetch result in cache.

        Args:
            url: Source URL
            fetch_result: Result from source fetcher

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled:
            return False

        # Only cache successful fetches with content
        if (fetch_result.get('availability_signal', 0) <= 0 or
            not fetch_result.get('extracted_text')):
            return False

        try:
            cache_key = self._generate_cache_key(url)
            ttl = self._determine_ttl(url)

            # Prepare cache payload
            cache_data = {
                'url': url,
                'normalized_url': self._normalize_url(url),
                'extracted_text': fetch_result['extracted_text'],
                'content_length': fetch_result['content_length'],
                'final_url': fetch_result.get('final_url', url),
                'fetch_timestamp': time.time(),
                'cached_timestamp': time.time(),
                'availability_signal': fetch_result['availability_signal'],
                'error_type': fetch_result.get('error_type'),
                'snapshot_hash': fetch_result.get('snapshot_hash'),
                'ttl': ttl
            }

            # Store in Redis with TTL
            cache_payload = json.dumps(cache_data, ensure_ascii=False)
            success = self.redis.setex(cache_key, ttl, cache_payload)

            if success:
                logger.info(f"💾 Cache STORE: {url} (TTL: {ttl}s)")
                return True
            else:
                logger.warning(f"⚠️ Cache store failed for {url}")
                return False

        except Exception as e:
            logger.error(f"Cache write error for {url}: {str(e)}")
            return False

    def delete(self, url: str) -> bool:
        """
        Delete cached entry for URL.

        Args:
            url: Source URL to remove from cache

        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled:
            return False

        try:
            cache_key = self._generate_cache_key(url)
            result = self.redis.delete(cache_key)
            if result:
                logger.info(f"🗑️ Cache DELETE: {url}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error for {url}: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and health info.

        Returns:
            Cache statistics dict
        """
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "reason": "Redis unavailable or module missing"
            }

        try:
            info = self.redis.info()
            keyspace = self.redis.info('keyspace')

            # Count ancestor keys
            ancestor_keys = 0
            try:
                keys = self.redis.keys('ancestor:source:v3:*')
                ancestor_keys = len(keys)
            except Exception:
                pass

            return {
                "enabled": True,
                "status": "online",
                "redis_version": info.get('redis_version'),
                "connected_clients": info.get('connected_clients'),
                "used_memory_human": info.get('used_memory_human'),
                "ancestor_cache_keys": ancestor_keys,
                "keyspace": keyspace
            }

        except Exception as e:
            return {
                "enabled": True,
                "status": "error",
                "error": str(e)
            }


# Convenience functions for backward compatibility
def create_cache_service(host='localhost', port=6379, password=None) -> CacheService:
    """Create a cache service instance."""
    return CacheService(host=host, port=port, password=password)


if __name__ == "__main__":
    # Test cache service functionality
    import sys

    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing Cache Service")
    print("=" * 50)

    # Test without Redis (should fail gracefully)
    cache = CacheService(host="nonexistent-redis-host")
    print(f"Cache enabled: {cache.enabled}")

    # Test URL normalization
    test_urls = [
        "https://www.example.com/path?utm_source=twitter&real_param=value",
        "https://Example.COM/path/?real_param=value&utm_campaign=test",
        "https://www.example.com/path/?real_param=value#fragment"
    ]

    print(f"\n🧪 URL Normalization Test:")
    for url in test_urls:
        normalized = cache._normalize_url(url)
        key = cache._generate_cache_key(url)
        print(f"Original:   {url}")
        print(f"Normalized: {normalized}")
        print(f"Cache Key:  {key[:20]}...")
        print("-" * 50)

    print(f"\n✅ Cache Service test complete")
