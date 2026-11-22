from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

# One shared limiter. `default_limits` apply to every route through SlowAPIMiddleware (see main.py);
# sensitive routes add a stricter `@limiter.limit(...)` of their own.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_PER_MINUTE],
    enabled=settings.ENVIRONMENT.lower() != "test",
)
