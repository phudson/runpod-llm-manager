"""
RunPod LLM Manager FastAPI Proxy
Modern service-layer architecture with dependency injection and comprehensive security
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .dependencies import Dependencies, get_default_dependencies
from .services import LLMService, HealthService, MetricsService
from .proxy_fastapi_models import ChatCompletionRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "llm-proxy"}'
)
logger = logging.getLogger(__name__)

# Global services (initialized on startup)
llm_service: Optional[LLMService] = None
health_service: Optional[HealthService] = None
metrics_service: Optional[MetricsService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global llm_service, health_service, metrics_service

    # Initialize services on startup
    deps = get_default_dependencies()
    llm_service = LLMService(deps)
    health_service = HealthService(deps)
    metrics_service = MetricsService(deps)

    logger.info("Services initialized successfully")
    yield
    logger.info("Application shutting down")

app = FastAPI(
    title="RunPod LLM Manager",
    description="Secure and performant LLM proxy with Model Store integration",
    version="2.0.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    deps = get_default_dependencies()

    if not deps.rate_limiter.is_allowed(client_ip):
        remaining = deps.rate_limiter.get_remaining_requests(client_ip)
        from datetime import datetime, timedelta
        reset_time = datetime.now() + timedelta(seconds=config.rate_limit_window)

        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {config.rate_limit_requests} per {config.rate_limit_window} seconds",
                "retry_after": int((reset_time - datetime.now()).total_seconds()),
                "remaining_requests": remaining
            },
            headers={
                "Retry-After": str(int((reset_time - datetime.now()).total_seconds())),
                "X-RateLimit-Limit": str(config.rate_limit_requests),
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
    if config.use_https:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

    return response

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

# API Routes

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint for monitoring proxy status."""
    if health_service is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    client_ip = request.client.host if request.client else "unknown"
    return await health_service.get_health_status(client_ip)

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    if metrics_service is None:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})
    return await metrics_service.get_metrics()

@app.get("/dashboard")
async def dashboard(request: Request):
    """Comprehensive dashboard with health, metrics, and system info."""
    if health_service is None or metrics_service is None:
        return JSONResponse(status_code=503, content={"error": "Services not initialized"})

    client_ip = request.client.host if request.client else "unknown"

    # Get health info
    health_info = await health_service.get_health_status(client_ip)

    # Get metrics
    metrics_info = await metrics_service.get_metrics()

    # System information
    import sys
    system_info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "pid": 0,  # Would need to get actual PID
        "working_directory": ""  # Would need to get actual working directory
    }

    return {
        "health": health_info,
        "metrics": metrics_info,
        "system": system_info,
        "configuration": {
            "runpod_endpoint": config.runpod_endpoint,
            "cache_dir": config.cache_dir,
            "max_cache_size": config.max_cache_size,
            "max_cache_bytes": config.cache_size_bytes,
            "profiling_enabled": False,  # Would need to add to config
            "security": {
                "rate_limiting_enabled": True,
                "input_validation_enabled": True,
                "security_headers_enabled": True,
                "https_enforced": config.use_https
            }
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Process chat completion requests."""
    if llm_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return await llm_service.process_completion_request(request)
    except Exception as e:
        # Log security-relevant errors
        client_ip = getattr(request, 'client', None)
        client_ip = client_ip.host if client_ip else "unknown"

        log_security_event("CHAT_COMPLETION_ERROR", {
            "client_ip": client_ip,
            "path": "/v1/chat/completions",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Streaming version for ultra-low latency responses."""
    # For now, delegate to regular handler
    # In a full implementation, this would return a streaming response
    return await chat_completions(request)

# Development endpoints
if config.test_mode or True:  # Enable for development
    @app.get("/debug/cache")
    async def debug_cache():
        """Debug endpoint to inspect cache contents (development only)."""
        # Placeholder - would need to implement cache inspection
        return {"message": "Cache debug not implemented in service layer"}

# Server startup
if __name__ == "__main__":
    import uvicorn

    uvicorn_config = {
        "app": app,
        "host": "0.0.0.0",
        "port": int(config.__dict__.get("port", 8000)),  # Would need to add to config
        "log_level": "info"
    }

    # Add SSL configuration if enabled
    if config.use_https and config.ssl_cert and config.ssl_key:
        uvicorn_config.update({
            "ssl_certfile": config.ssl_cert,
            "ssl_keyfile": config.ssl_key
        })
        logger.info(f"Starting HTTPS server with SSL cert: {config.ssl_cert}")
    else:
        logger.info("Starting HTTP server (no SSL)")

    uvicorn.run(**uvicorn_config)
