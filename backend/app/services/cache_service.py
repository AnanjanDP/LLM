import json
import hashlib
from typing import Optional, Dict, Any, Tuple
from app.core.config import settings
from app.core.logging import logger

try:
    import redis
except ImportError:
    redis = None


class RedisCacheService:
    """Production Redis caching service for queries, embeddings, and context with statistics tracking."""

    def __init__(self):
        self.redis_client = None
        self.hits = 0
        self.misses = 0
        self.enabled = settings.CACHE_ENABLED
        self._init_redis()

    def _init_redis(self):
        """Attempt connection to Redis server."""
        if not self.enabled or redis is None:
            logger.info("Redis cache disabled or redis-py not installed. Operating in-memory cache fallback mode.")
            return

        try:
            client = redis.Redis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_timeout=2.0
            )
            client.ping()
            self.redis_client = client
            logger.info(f"Connected to Redis cache at {settings.REDIS_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis server ({e}). Operating without Redis cache.")
            self.redis_client = None

    def _hash_key(self, prefix: str, key_str: str) -> str:
        """Create deterministic SHA256 cache key."""
        hashed = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        return f"rag:{prefix}:{hashed}"

    def get_query_cache(self, query: str, mode: str, model: str, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached RAG response for query + parameters + session."""
        if not self.redis_client:
            return None

        cache_key = self._hash_key("query", f"{query}:{mode}:{model}:{session_id or 'default'}")
        try:
            val = self.redis_client.get(cache_key)
            if val:
                self.hits += 1
                logger.info(f"Redis Cache HIT for key {cache_key[:20]}...")
                return json.loads(val)
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.misses += 1
            return None

    def set_query_cache(
        self, query: str, mode: str, model: str, data: Dict[str, Any], ttl: int = None, session_id: str = None
    ) -> bool:
        """Store RAG response in Redis with TTL."""
        if not self.redis_client:
            return False

        cache_key = self._hash_key("query", f"{query}:{mode}:{model}:{session_id or 'default'}")
        ttl_seconds = ttl or settings.CACHE_TTL_SECONDS
        try:
            self.redis_client.setex(cache_key, ttl_seconds, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def clear_cache(self) -> bool:
        """Flush query cache."""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Cleared Redis query cache.")
                return True
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")
                return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance metrics."""
        total = self.hits + self.misses
        hit_rate = round((self.hits / total * 100), 2) if total > 0 else 0.0
        return {
            "enabled": self.enabled and (self.redis_client is not None),
            "redis_connected": self.redis_client is not None,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": hit_rate,
        }


cache_service = RedisCacheService()

