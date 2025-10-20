"""
Health Service
Provides health check operations and system status monitoring.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import psutil

from .dependencies import Dependencies

logger = logging.getLogger(__name__)


class HealthService:
    """Service for health check operations."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.start_time = time.time()

    async def get_health_status(self, client_ip: str = "unknown") -> Dict[str, Any]:
        """Get comprehensive health status."""
        # Get cache statistics
        cache_stats = await self.deps.cache.get_stats()

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time

        # Get system resource usage
        system_stats = self._get_system_stats()

        # Check service dependencies
        dependency_status = await self._check_dependencies()

        remaining_requests = self.deps.rate_limiter.get_remaining_requests(client_ip)

        # Determine overall health status
        overall_status = "healthy"
        if not dependency_status["all_healthy"]:
            overall_status = "degraded"
        if system_stats["memory_percent"] > 90 or system_stats["cpu_percent"] > 95:
            overall_status = "critical"

        return {
            "status": overall_status,
            "timestamp": datetime.now().timestamp(),
            "uptime_seconds": uptime_seconds,
            "cache_dir": self.deps.config.cache_dir,
            "endpoint": self.deps.config.runpod_endpoint,
            "cache": {
                "entries": cache_stats["entries"],
                "size_bytes": cache_stats["total_size_bytes"],
                "max_entries": self.deps.config.max_cache_size,
                "max_bytes": self.deps.config.cache_size_bytes,
                "recent_accesses": cache_stats["recent_accesses"],
                "oldest_entry_age_seconds": cache_stats["oldest_entry_age"],
                "newest_entry_age_seconds": cache_stats["newest_entry_age"],
            },
            "system": system_stats,
            "dependencies": dependency_status,
            "security": {
                "rate_limit_remaining": remaining_requests,
                "rate_limit_limit": self.deps.config.rate_limit_requests,
                "rate_limit_window_seconds": self.deps.config.rate_limit_window,
                "https_enabled": self.deps.config.use_https,
                "cors_enabled": True,
            },
        }

    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_percent": psutil.disk_usage("/").percent,
                "disk_used_gb": psutil.disk_usage("/").used / (1024**3),
                "disk_total_gb": psutil.disk_usage("/").total / (1024**3),
                "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
                "process_count": len(psutil.pids()),
            }
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_used_gb": 0,
                "memory_total_gb": 0,
                "disk_percent": 0,
                "disk_used_gb": 0,
                "disk_total_gb": 0,
                "load_average": None,
                "process_count": 0,
                "error": str(e),
            }

    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check health of service dependencies."""
        results = {
            "cache": True,
            "rate_limiter": True,
            "filesystem": True,
            "config": True,
            "all_healthy": True,
        }

        # Test cache
        try:
            test_key = f"health_check_{int(time.time())}"
            await self.deps.cache.set(test_key, {"test": True})
            result = await self.deps.cache.get(test_key)
            results["cache"] = result is not None
            await self.deps.cache.delete(test_key)
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            results["cache"] = False

        # Test filesystem
        try:
            test_file = f"/tmp/health_check_{int(time.time())}.tmp"
            await self.deps.filesystem.write_file(test_file, "test")
            content = await self.deps.filesystem.read_file(test_file)
            results["filesystem"] = content == "test"
            await self.deps.filesystem.delete_file(test_file)
        except Exception as e:
            logger.warning(f"Filesystem health check failed: {e}")
            results["filesystem"] = False

        # Test rate limiter
        try:
            test_ip = f"health_check_{int(time.time())}"
            results["rate_limiter"] = self.deps.rate_limiter.is_allowed(test_ip)
        except Exception as e:
            logger.warning(f"Rate limiter health check failed: {e}")
            results["rate_limiter"] = False

        # Check config
        try:
            results["config"] = bool(self.deps.config.runpod_api_key)
        except Exception as e:
            logger.warning(f"Config health check failed: {e}")
            results["config"] = False

        results["all_healthy"] = all(results.values())
        return results
