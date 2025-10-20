"""
Metrics Service
Handles metrics collection and reporting for monitoring and analytics.
"""

import time
from collections import defaultdict
from typing import Any, Dict

from .dependencies import Dependencies


class MetricsService:
    """Service for metrics collection and reporting."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.response_times = []
        self.endpoint_metrics = defaultdict(int)
        self.client_metrics = defaultdict(int)
        self.start_time = time.time()

    def record_request(
        self, endpoint: str, client_ip: str, response_time: float, status_code: int = 200
    ):
        """Record a request metric."""
        self.request_count += 1
        self.response_times.append(response_time)
        self.endpoint_metrics[endpoint] += 1
        self.client_metrics[client_ip] += 1

        if status_code >= 400:
            self.error_count += 1

        # Keep only last 1000 response times for memory efficiency
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    async def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics."""
        cache_stats = await self.deps.cache.get_stats()

        # Calculate averages
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        )

        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_requests) if total_cache_requests > 0 else 0.0
        )

        # Calculate error rate
        error_rate = (self.error_count / self.request_count) if self.request_count > 0 else 0.0

        # Calculate requests per second
        uptime_seconds = time.time() - self.start_time
        requests_per_second = self.request_count / uptime_seconds if uptime_seconds > 0 else 0

        return {
            "requests_total": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "errors_total": self.error_count,
            "avg_response_time": avg_response_time,
            "cache_entries": cache_stats["entries"],
            "cache_size_bytes": cache_stats["total_size_bytes"],
            "cache_hit_rate": cache_hit_rate,
            "error_rate": error_rate,
            "requests_per_second": requests_per_second,
            "uptime_seconds": uptime_seconds,
            "endpoint_breakdown": dict(self.endpoint_metrics),
            "top_clients": dict(
                sorted(self.client_metrics.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "rate_limiter_stats": {
                "current_limit": self.deps.config.rate_limit_requests,
                "window_seconds": self.deps.config.rate_limit_window,
            },
        }
