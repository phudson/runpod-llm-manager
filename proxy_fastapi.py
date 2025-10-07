# Standard library imports
import os
import json
import hashlib
import logging
import time
import sys
import asyncio
import re
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Annotated

# Third-party imports
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import aiofiles
from pydantic import BaseModel, Field, validator


# Configuration and constants
class Config:
    """Centralized configuration management."""

    # API settings
    RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/llm_cache")

    # Security settings
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "50000"))  # 50KB

    # Cache settings
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    CACHE_SIZE_BYTES = int(os.getenv("CACHE_SIZE_BYTES", "1073741824"))  # 1GB

    # Feature flags
    ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"

    # SSL settings
    SSL_CERT = os.getenv("SSL_CERT")
    SSL_KEY = os.getenv("SSL_KEY")

    # CORS settings - restrict to known safe origins
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080").split(",")


# Global state (consider using dependency injection for better testability)
rate_limit_store = {}
backend_client = None

# Threading locks
cache_lock = threading.Lock()
index_lock = threading.Lock()
metrics_lock = threading.Lock()

# Cache data structures
cache_metadata = OrderedDict()  # LRU tracking: key -> (timestamp, size)
cache_index = {}  # key -> cache_path for O(1) lookups

# Metrics
metrics = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "errors_total": 0,
    "avg_response_time": 0.0,
    "total_response_time": 0.0
}

# Pydantic models for input validation
class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r'^(user|assistant|system)$')
    content: str = Field(..., min_length=1, max_length=10000)

    @validator('content')
    def sanitize_content(cls, v):
        # Basic sanitization - remove potentially harmful patterns
        v = re.sub(r'<script[^>]*>.*?</script>', '', v, flags=re.IGNORECASE)
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        return v

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=100)
    messages: Annotated[List[ChatMessage], Field(min_length=1, max_length=50)]
    max_tokens: int = Field(10, ge=1, le=4096)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    stream: bool = Field(False)

class RateLimiter:
    """In-memory rate limiter with sliding window.
    Note: Rate limits reset on application restart.
    For production, consider Redis or database-backed rate limiting.
    """

    @staticmethod
    def is_allowed(client_ip: str) -> bool:
        """Check if request is allowed under rate limit."""
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=Config.RATE_LIMIT_WINDOW)

            if client_ip not in rate_limit_store:
                rate_limit_store[client_ip] = []

            # Clean old requests outside the window
            rate_limit_store[client_ip] = [
                req_time for req_time in rate_limit_store[client_ip]
                if req_time > window_start
            ]

            # Check if under limit
            if len(rate_limit_store[client_ip]) < Config.RATE_LIMIT_REQUESTS:
                rate_limit_store[client_ip].append(now)
                return True

            return False
        except Exception as e:
            # Fail open on rate limiter errors to avoid blocking legitimate traffic
            logging.warning(f"Rate limiter error for {client_ip}: {e}")
            return True

    @staticmethod
    def get_remaining_requests(client_ip: str) -> int:
        """Get remaining requests in current window."""
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=Config.RATE_LIMIT_WINDOW)

            if client_ip not in rate_limit_store:
                return Config.RATE_LIMIT_REQUESTS

            # Clean old requests
            rate_limit_store[client_ip] = [
                req_time for req_time in rate_limit_store[client_ip]
                if req_time > window_start
            ]

            return max(0, Config.RATE_LIMIT_REQUESTS - len(rate_limit_store[client_ip]))
        except Exception as e:
            logging.warning(f"Error getting remaining requests for {client_ip}: {e}")
            return Config.RATE_LIMIT_REQUESTS

app = FastAPI()
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"

    if not RateLimiter.is_allowed(client_ip):
        remaining = RateLimiter.get_remaining_requests(client_ip)
        reset_time = datetime.now() + timedelta(seconds=Config.RATE_LIMIT_WINDOW)

        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {Config.RATE_LIMIT_REQUESTS} per {Config.RATE_LIMIT_WINDOW} seconds",
                "retry_after": int((reset_time - datetime.now()).total_seconds()),
                "remaining_requests": remaining
            },
            headers={
                "Retry-After": str(int((reset_time - datetime.now()).total_seconds())),
                "X-RateLimit-Limit": str(Config.RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(reset_time.timestamp()))
            }
        )

    response = await call_next(request)
    return response

# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # HTTPS enforcement (HSTS) - only if HTTPS is enabled
    if Config.USE_HTTPS:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

    return response

# Cache management - optimized for speed
# Note: Global state - consider dependency injection for better testability

# Metrics
metrics = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "errors_total": 0,
    "avg_response_time": 0.0,
    "total_response_time": 0.0
}
metrics_lock = threading.Lock()

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "llm-proxy"}'
)
logger = logging.getLogger(__name__)

# Security event logging
def log_security_event(event_type: str, details: dict, level="WARNING"):
    """Log security-related events with structured data."""
    logger.log(getattr(logging, level), f"SECURITY: {event_type}", extra={
        "event_type": event_type,
        "client_ip": details.get("client_ip", "unknown"),
        "user_agent": details.get("user_agent", "unknown"),
        "request_path": details.get("path", "unknown"),
        "suspicious_pattern": details.get("pattern", ""),
        "request_size": details.get("request_size", 0),
        "rate_limit_remaining": details.get("rate_limit_remaining", 0)
    })

# Track application start time
START_TIME = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize cache index on startup for faster lookups."""
    logger.info("Initializing cache index...")
    start_init = time.time()

    # Build fast lookup index from existing cache files
    with index_lock:
        cache_index.clear()
        if os.path.exists(Config.CACHE_DIR):
            for filename in os.listdir(Config.CACHE_DIR):
                if filename.endswith('.json'):
                    key = filename[:-5]  # Remove .json extension
                    cache_path = os.path.join(Config.CACHE_DIR, filename)
                    cache_index[key] = cache_path

    init_time = time.time() - start_init
    logger.info(f"Cache index initialized with {len(cache_index)} entries in {init_time:.3f}s")

    # Initialize global HTTP client
    global backend_client
    backend_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )

    # Pre-warm cache with common patterns if enabled
    if os.getenv("PREWARM_CACHE", "false").lower() == "true":
        asyncio.create_task(prewarm_cache())

async def prewarm_cache():
    """Pre-warm cache with common code patterns for faster Kilo Code responses."""
    common_patterns = [
        {"messages": [{"role": "user", "content": "def "}], "max_tokens": 50},
        {"messages": [{"role": "user", "content": "class "}], "max_tokens": 50},
        {"messages": [{"role": "user", "content": "import "}], "max_tokens": 30},
        {"messages": [{"role": "user", "content": "function "}], "max_tokens": 50},
        {"messages": [{"role": "user", "content": "const "}], "max_tokens": 40},
        {"messages": [{"role": "user", "content": "print("}], "max_tokens": 20},
    ]

    logger.info("Pre-warming cache with common patterns...")

    for pattern in common_patterns:
        try:
            key = get_cache_key(pattern)
            cache_path = os.path.join(CACHE_DIR, key + ".json")

            # Skip if already cached
            if os.path.exists(cache_path):
                continue

            # Make request to backend to populate cache
            if backend_client:
                response = await backend_client.post(Config.RUNPOD_ENDPOINT, json=pattern)
                if response.status_code == 200:
                    result = response.json()
                    await save_cache_async(cache_path, result)
                    update_cache_index(key, cache_path)

                    cache_size = len(json.dumps(result, separators=(',', ':')))
                    update_cache_metadata(key, cache_size)

                    logger.debug(f"Pre-warmed cache for pattern: {pattern['messages'][0]['content'][:10]}...")

        except Exception as e:
            logger.debug(f"Failed to pre-warm pattern {pattern}: {e}")
            continue

    logger.info("Cache pre-warming completed")

def get_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cache_size() -> int:
    """Get current cache size in bytes."""
    total_size = 0
    for filename in os.listdir(Config.CACHE_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(Config.CACHE_DIR, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                pass
    return total_size

def evict_cache_if_needed():
    """Evict least recently used cache entries if cache limits are exceeded."""
    with cache_lock:
        # Check size limits
        current_size = len(cache_metadata)
        current_bytes = get_cache_size()

        while (current_size > Config.MAX_CACHE_SIZE or current_bytes > Config.CACHE_SIZE_BYTES) and cache_metadata:
            # Remove oldest entry (LRU)
            key, (timestamp, size) = cache_metadata.popitem(last=False)
            cache_path = os.path.join(Config.CACHE_DIR, key + ".json")
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    current_size -= 1
                    current_bytes -= size
                logger.info(f"Evicted cache entry: {key}")
            except OSError as e:
                logger.error(f"Failed to evict cache entry {key}: {e}")

def update_cache_metadata(key: str, size: int):
    """Update cache metadata for LRU tracking."""
    with cache_lock:
        current_time = time.time()
        cache_metadata[key] = (current_time, size)
        cache_metadata.move_to_end(key)  # Mark as most recently used

def update_cache_index(key: str, cache_path: str):
    """Update fast cache index for O(1) lookups."""
    with index_lock:
        cache_index[key] = cache_path

def get_cached_response_fast(key: str):
    """Fast cache lookup using in-memory index."""
    with index_lock:
        cache_path = cache_index.get(key)
        if not cache_path or not os.path.exists(cache_path):
            return None

    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # Remove corrupted cache entry
        with index_lock:
            cache_index.pop(key, None)
        return None

async def save_cache_async(cache_path: str, data: dict):
    """Async cache saving for better performance."""
    try:
        async with aiofiles.open(cache_path, 'w') as f:
            await f.write(json.dumps(data, separators=(',', ':')))  # Compact JSON
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def record_metric(name: str, value: float = 1.0):
    """Record a metric in a thread-safe way."""
    with metrics_lock:
        if name in metrics:
            if name == "avg_response_time":
                # Special handling for average calculation
                metrics["total_response_time"] += value
                metrics["requests_total"] += 1
                metrics["avg_response_time"] = metrics["total_response_time"] / metrics["requests_total"]
            else:
                metrics[name] += value

def get_health_data(client_ip: str = "unknown"):
    """Get health data for both health endpoint and dashboard."""
    cache_size = get_cache_size()
    cache_count = len([f for f in os.listdir(Config.CACHE_DIR) if f.endswith('.json')])
    remaining_requests = RateLimiter.get_remaining_requests(client_ip)

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_dir": Config.CACHE_DIR,
        "endpoint": Config.RUNPOD_ENDPOINT,
        "cache_entries": cache_count,
        "cache_size_bytes": cache_size,
        "max_cache_size": Config.MAX_CACHE_SIZE,
        "max_cache_bytes": Config.CACHE_SIZE_BYTES,
        "uptime": time.time() - START_TIME,
        "security": {
            "rate_limit_remaining": remaining_requests,
            "rate_limit_limit": Config.RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": Config.RATE_LIMIT_WINDOW,
            "https_enabled": Config.USE_HTTPS,
            "cors_enabled": True
        }
    }

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint for monitoring proxy status."""
    client_ip = request.client.host if request.client else "unknown"
    return get_health_data(client_ip)

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    with metrics_lock:
        current_metrics = metrics.copy()

    # Add cache metrics
    cache_size = get_cache_size()
    cache_count = len([f for f in os.listdir(Config.CACHE_DIR) if f.endswith('.json')])

    current_metrics.update({
        "cache_entries": cache_count,
        "cache_size_bytes": cache_size,
        "cache_hit_rate": (current_metrics["cache_hits"] / max(current_metrics["requests_total"], 1)) * 100,
        "error_rate": (current_metrics["errors_total"] / max(current_metrics["requests_total"], 1)) * 100
    })

    return current_metrics

@app.get("/dashboard")
async def dashboard(request: Request):
    """Comprehensive dashboard with health, metrics, and system info."""
    client_ip = request.client.host if request.client else "unknown"

    # Get health info
    health_info = get_health_data(client_ip)

    # Get metrics
    metrics_info = await get_metrics()

    # System information
    system_info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "pid": os.getpid(),
        "working_directory": os.getcwd()
    }

    return {
        "health": health_info,
        "metrics": metrics_info,
        "system": system_info,
        "configuration": {
            "runpod_endpoint": Config.RUNPOD_ENDPOINT,
            "cache_dir": Config.CACHE_DIR,
            "max_cache_size": Config.MAX_CACHE_SIZE,
            "max_cache_bytes": Config.CACHE_SIZE_BYTES,
            "profiling_enabled": Config.ENABLE_PROFILING,
            "security": {
                "rate_limiting_enabled": True,
                "input_validation_enabled": True,
                "security_headers_enabled": True,
                "https_enforced": Config.USE_HTTPS
            }
        }
    }

@app.get("/debug/cache")
async def debug_cache():
    """Debug endpoint to inspect cache contents (development only)."""
    if not Config.ENABLE_PROFILING:
        return {"error": "Profiling not enabled"}

    cache_info = {}
    for filename in os.listdir(Config.CACHE_DIR):
        if filename.endswith('.json'):
            key = filename[:-5]  # Remove .json extension
            filepath = os.path.join(Config.CACHE_DIR, filename)
            try:
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                cache_info[key] = {
                    "size": size,
                    "modified": mtime,
                    "age_seconds": time.time() - mtime
                }
            except OSError:
                pass

    return {
        "cache_entries": cache_info,
        "total_entries": len(cache_info),
        "lru_order": list(cache_metadata.keys())[-10:]  # Last 10 entries
    }

@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: Request):
    """Streaming version for ultra-low latency responses."""
    return await chat_completions(request)  # For now, delegate to regular handler

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start_time = time.time()
    record_metric("requests_total")

    # Extract client information for security logging
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "unknown")
    content_length = int(request.headers.get("Content-Length", 0))

    try:
        # Validate and sanitize input
        payload_dict = await request.json()

        # Security checks
        if content_length > Config.MAX_REQUEST_SIZE:  # Configurable limit
            log_security_event("REQUEST_TOO_LARGE", {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "path": "/v1/chat/completions",
                "request_size": content_length
            })
            raise HTTPException(status_code=413, detail="Request too large")

        # Validate with Pydantic model
        try:
            validated_payload = ChatCompletionRequest(**payload_dict)
            payload = validated_payload.dict()
        except Exception as e:
            log_security_event("INVALID_INPUT", {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "path": "/v1/chat/completions",
                "pattern": str(e)
            })
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

        # Additional security checks
        total_content_length = sum(len(msg['content']) for msg in payload['messages'])
        if total_content_length > Config.MAX_CONTENT_LENGTH:  # Configurable content limit
            log_security_event("CONTENT_TOO_LONG", {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "path": "/v1/chat/completions",
                "request_size": total_content_length
            })
            raise HTTPException(status_code=400, detail="Content too long")

        key = get_cache_key(payload)
        cache_path = os.path.join(Config.CACHE_DIR, key + ".json")

        # Ultra-fast cache lookup using in-memory index
        cached_result = get_cached_response_fast(key)
        if cached_result is not None:
            # Update LRU metadata
            cache_size = len(json.dumps(cached_result, separators=(',', ':')))
            update_cache_metadata(key, cache_size)

            record_metric("cache_hits")
            response_time = time.time() - start_time
            record_metric("avg_response_time", response_time)

            logger.info(f"Cache hit for key: {key[:8]}...", extra={
                "cache_hit": True,
                "response_time": response_time,
                "cache_size": cache_size
            })

            return JSONResponse(content=cached_result)

        # Cache miss - fetch from backend
        record_metric("cache_misses")
        logger.info(f"Cache miss for key: {key[:8]}...", extra={
            "cache_hit": False
        })

        # Use persistent HTTP client for better performance
        global backend_client
        if backend_client is None:
            backend_client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )

        async with backend_client:
            response = await backend_client.post(Config.RUNPOD_ENDPOINT, json=payload)
            response.raise_for_status()
            result = response.json()

        # Async cache saving for better performance
        await save_cache_async(cache_path, result)

        # Update cache index for fast lookups
        update_cache_index(key, cache_path)

        # Update cache metadata and evict if needed
        cache_size = len(json.dumps(result, separators=(',', ':')))
        update_cache_metadata(key, cache_size)
        evict_cache_if_needed()

        response_time = time.time() - start_time
        record_metric("avg_response_time", response_time)

        logger.info(f"Cached new response for key: {key[:8]}...", extra={
            "response_time": response_time,
            "cache_size": cache_size,
            "backend_status": response.status_code
        })

        return JSONResponse(content=result)

    except Exception as e:
        record_metric("errors_total")
        response_time = time.time() - start_time

        # Log security-relevant errors
        if isinstance(e, HTTPException):
            # Client errors - log at info level
            logger.info(f"Client error: {str(e)}", extra={
                "error": str(e),
                "response_time": response_time,
                "status_code": e.status_code,
                "client_ip": client_ip
            })
        else:
            # Server errors - log security event
            log_security_event("SERVER_ERROR", {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "path": "/v1/chat/completions",
                "error": str(e)
            })
            logger.error(f"Server error processing request: {str(e)}", extra={
                "error": str(e),
                "response_time": response_time,
                "client_ip": client_ip
            })
        raise

# SSL/TLS server configuration
if __name__ == "__main__":
    import uvicorn

    # Default configuration
    uvicorn_config = {
        "app": app,
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", "8000")),
        "log_level": "info"
    }

    # Add SSL configuration if enabled
    if Config.USE_HTTPS and Config.SSL_CERT and Config.SSL_KEY:
        uvicorn_config.update({
            "ssl_certfile": Config.SSL_CERT,
            "ssl_keyfile": Config.SSL_KEY
        })
        logger.info(f"Starting HTTPS server with SSL cert: {Config.SSL_CERT}")
    else:
        logger.info("Starting HTTP server (no SSL)")

    uvicorn.run(**uvicorn_config)
