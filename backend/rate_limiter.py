import time
from fastapi import HTTPException
from functools import wraps

# Setup to import redis_client from backend.state to avoid circular deps with main
# But this file imports from it, so main shouldn't import this at top level if this imports main
# In this refactor, we import redis_client from state.

def simple_rate_limit(redis_client, key: str, limit: int, window_sec: int):
    """
    Baby-level rate limiter:
    - key: a string to identify client (IP or endpoint)
    - limit: max requests per window
    - window_sec: time window in seconds
    """
    now = int(time.time())
    window_key = f"rate_limit:{key}:{now // window_sec}"

    count = redis_client.incr(window_key)
    redis_client.expire(window_key, window_sec)
    
    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ============================================================
# DECORATOR WRAPPER FOR FASTAPI ENDPOINTS
# ============================================================

def rate_limit(limit: int, window_sec: int, key_prefix: str = ""):
    """
    Decorator form of simple_rate_limit(), without modifying its code.

    - Works on async FastAPI endpoints.
    - Supports per-ticker rate limiting when the request body contains "ticker".
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Late import to avoid circular dependencies if necessary, 
            # or import from state.
            from backend.state import redis_client

            # Only apply limiter if Redis is available
            if redis_client:
                try:
                    key = key_prefix

                    # If endpoint receives request body, extract ticker for per-symbol limiting
                    if "request" in kwargs:
                        try:
                            req_data = await kwargs["request"].json()
                            if isinstance(req_data, dict) and "ticker" in req_data:
                                key = f"{key_prefix}:{req_data['ticker'].strip().upper()}"
                        except Exception:
                            pass

                    simple_rate_limit(redis_client, key=key, limit=limit, window_sec=window_sec)

                except HTTPException:
                    # Re-raise to let FastAPI return 429
                    raise

            return await func(*args, **kwargs)
        return wrapper
    return decorator
