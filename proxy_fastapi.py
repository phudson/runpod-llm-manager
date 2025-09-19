import os, json, hashlib, httpx, logging, time, sys, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from collections import OrderedDict
from typing import Dict, Any
import threading
import aiofiles

app = FastAPI()
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "http://localhost:8000/v1/chat/completions")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache management - optimized for speed
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))  # Max number of cached responses
CACHE_SIZE_BYTES = int(os.getenv("CACHE_SIZE_BYTES", "1073741824"))  # 1GB default
cache_metadata = OrderedDict()  # LRU tracking: key -> (timestamp, size)
cache_lock = threading.Lock()

# Fast cache lookup - in-memory index
cache_index = {}  # key -> cache_path for O(1) lookups
index_lock = threading.Lock()

# Connection pool for backend requests
backend_client = None

# Performance profiling
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"

# SSL/TLS configuration
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"
SSL_CERT = os.getenv("SSL_CERT")
SSL_KEY = os.getenv("SSL_KEY")

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
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.json'):
                    key = filename[:-5]  # Remove .json extension
                    cache_path = os.path.join(CACHE_DIR, filename)
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
                response = await backend_client.post(RUNPOD_ENDPOINT, json=pattern)
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
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(CACHE_DIR, filename)
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

        while (current_size > MAX_CACHE_SIZE or current_bytes > CACHE_SIZE_BYTES) and cache_metadata:
            # Remove oldest entry (LRU)
            key, (timestamp, size) = cache_metadata.popitem(last=False)
            cache_path = os.path.join(CACHE_DIR, key + ".json")
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

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring proxy status."""
    cache_size = get_cache_size()
    cache_count = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_dir": CACHE_DIR,
        "endpoint": RUNPOD_ENDPOINT,
        "cache_entries": cache_count,
        "cache_size_bytes": cache_size,
        "max_cache_size": MAX_CACHE_SIZE,
        "max_cache_bytes": CACHE_SIZE_BYTES,
        "uptime": time.time() - START_TIME
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    with metrics_lock:
        current_metrics = metrics.copy()

    # Add cache metrics
    cache_size = get_cache_size()
    cache_count = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

    current_metrics.update({
        "cache_entries": cache_count,
        "cache_size_bytes": cache_size,
        "cache_hit_rate": (current_metrics["cache_hits"] / max(current_metrics["requests_total"], 1)) * 100,
        "error_rate": (current_metrics["errors_total"] / max(current_metrics["requests_total"], 1)) * 100
    })

    return current_metrics

@app.get("/dashboard")
async def dashboard():
    """Comprehensive dashboard with health, metrics, and system info."""
    # Get health info
    health_info = await health_check()

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
            "runpod_endpoint": RUNPOD_ENDPOINT,
            "cache_dir": CACHE_DIR,
            "max_cache_size": MAX_CACHE_SIZE,
            "max_cache_bytes": CACHE_SIZE_BYTES,
            "profiling_enabled": ENABLE_PROFILING
        }
    }

@app.get("/debug/cache")
async def debug_cache():
    """Debug endpoint to inspect cache contents (development only)."""
    if not ENABLE_PROFILING:
        return {"error": "Profiling not enabled"}

    cache_info = {}
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith('.json'):
            key = filename[:-5]  # Remove .json extension
            filepath = os.path.join(CACHE_DIR, filename)
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

    try:
        payload = await request.json()
        key = get_cache_key(payload)
        cache_path = os.path.join(CACHE_DIR, key + ".json")

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
            response = await backend_client.post(RUNPOD_ENDPOINT, json=payload)
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
        logger.error(f"Error processing request: {str(e)}", extra={
            "error": str(e),
            "response_time": response_time
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
    if USE_HTTPS and SSL_CERT and SSL_KEY:
        uvicorn_config.update({
            "ssl_certfile": SSL_CERT,
            "ssl_keyfile": SSL_KEY
        })
        logger.info(f"Starting HTTPS server with SSL cert: {SSL_CERT}")
    else:
        logger.info("Starting HTTP server (no SSL)")

    uvicorn.run(**uvicorn_config)
