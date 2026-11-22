import json
import hashlib
from typing import Optional, Dict, Any
from app.core.config import settings
from app.core.logging import logger

try:
    import redis
except ImportError:
    redis = None

KEY_PREFIX = "rag:"


class RedisCacheService:
    """Redis response cache for chat queries, with hit/miss statistics.

    The cache key covers everything that can change the answer: query, retrieval mode, model, provider,
    prompt version, top-k, reranking, the conversation, and the version of the document index. Leaving any of
    them out would serve one setting's answer to another, or serve an answer computed from documents that
    have since been replaced.
    """

    def __init__(self):
        self.redis_client = None
        self.hits = 0
        self.misses = 0
        self.enabled = settings.CACHE_ENABLED
        self._init_redis()

    def _init_redis(self):
        """Attempt connection to Redis server."""
        if not self.enabled or redis is None:
            logger.info("Redis cache disabled or redis-py not installed. Responses will not be cached.")
            return

        try:
            client = redis.Redis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_timeout=2.0
            )
            client.ping()
            self.redis_client = client
            logger.info("Connected to Redis cache.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis server ({e}). Operating without Redis cache.")
            self.redis_client = None

    def _key(self, **fields: Any) -> str:
        """Deterministic SHA256 cache key over all fields."""
        canonical = json.dumps(fields, sort_keys=True, default=str)
        return f"{KEY_PREFIX}query:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    def get_query_cache(self, **fields: Any) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response for exactly these parameters."""
        if not self.redis_client:
            return None

        key = self._key(**fields)
        try:
            val = self.redis_client.get(key)
            if val:
                self.hits += 1
                return json.loads(val)
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.misses += 1
            return None

    def set_query_cache(self, data: Dict[str, Any], ttl: int = None, **fields: Any) -> bool:
        """Store a response in Redis with TTL."""
        if not self.redis_client:
            return False

        try:
            self.redis_client.setex(self._key(**fields), ttl or settings.CACHE_TTL_SECONDS, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def clear_cache(self) -> bool:
        """Delete this application's cache entries only (never the whole Redis database)."""
        if not self.redis_client:
            return True
        try:
            for key in self.redis_client.scan_iter(match=f"{KEY_PREFIX}*", count=500):
                self.redis_client.delete(key)
            logger.info("Cleared RAG response cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance metrics."""
        total = self.hits + self.misses
        return {
            "enabled": self.enabled and (self.redis_client is not None),
            "redis_connected": self.redis_client is not None,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": round((self.hits / total * 100), 2) if total > 0 else 0.0,
        }


cache_service = RedisCacheService()
