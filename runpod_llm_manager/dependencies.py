"""
Dependency injection framework for RunPod LLM Manager
Provides protocols and dependency containers for testable architecture
"""

from dataclasses import dataclass

# ABC imports removed - not used in this file
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol

import httpx

from .config import AppConfig


class HTTPClientProtocol(Protocol):
    """Protocol for HTTP client operations."""

    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make POST request and return JSON response."""
        ...

    async def post_stream(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Make POST request and return streaming response."""
        # This will be properly implemented by concrete classes
        if False:  # pragma: no cover
            yield ""  # Make type checker happy
        raise NotImplementedError("post_stream must be implemented by concrete HTTP client")

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make GET request and return JSON response."""
        ...

    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make DELETE request and return JSON response."""
        ...


class CacheProtocol(Protocol):
    """Protocol for cache operations."""

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key."""
        ...

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set cached value."""
        ...

    async def delete(self, key: str) -> None:
        """Delete cached value."""
        ...

    async def clear(self) -> None:
        """Clear all cached values."""
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


class FileSystemProtocol(Protocol):
    """Protocol for file system operations."""

    async def write_file(self, path: str, content: str) -> None:
        """Write content to file asynchronously."""
        ...

    async def read_file(self, path: str) -> str:
        """Read content from file asynchronously."""
        ...

    async def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        ...

    async def delete_file(self, path: str) -> None:
        """Delete file."""
        ...


class RateLimiterProtocol(Protocol):
    """Protocol for rate limiting operations."""

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed under rate limit."""
        ...

    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests in current window."""
        ...


@dataclass
class Dependencies:
    """Container for all application dependencies."""

    config: AppConfig
    http_client: HTTPClientProtocol
    cache: CacheProtocol
    filesystem: FileSystemProtocol
    rate_limiter: RateLimiterProtocol


# Concrete implementations


class HTTPXClient:
    """HTTPX-based HTTP client implementation."""

    def __init__(self, timeout: float = 30.0, base_headers: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.base_headers = base_headers or {}

    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        request_headers = {**self.base_headers}
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=json, headers=request_headers, **kwargs)
            response.raise_for_status()
            return response.json()

    async def post_stream(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Make POST request and return streaming response."""
        request_headers = {**self.base_headers}
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, json=json, headers=request_headers, **kwargs
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, **kwargs)
            response.raise_for_status()
            return response.json()

    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}


class InMemoryCache:
    """In-memory cache implementation."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        import time

        self._time = time.time

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._store:
            self._access_times[key] = self._time()
            return self._store[key]
        return None

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        self._store[key] = value
        self._access_times[key] = self._time()

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._access_times.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()
        self._access_times.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = self._time()
        total_size = sum(len(str(v).encode("utf-8")) for v in self._store.values())

        # Calculate hit rate (simplified - would need hit/miss counters in real implementation)
        recent_accesses = sum(1 for t in self._access_times.values() if now - t < 3600)  # Last hour

        return {
            "entries": len(self._store),
            "total_size_bytes": total_size,
            "recent_accesses": recent_accesses,
            "oldest_entry_age": min((now - t for t in self._access_times.values()), default=0),
            "newest_entry_age": max((now - t for t in self._access_times.values()), default=0),
        }


class AIOFilesFileSystem:
    """aiofiles-based file system implementation."""

    def __init__(self):
        try:
            import aiofiles

            self.aiofiles = aiofiles
        except ImportError:
            raise ImportError("aiofiles is required for file system operations")

    async def write_file(self, path: str, content: str) -> None:
        async with self.aiofiles.open(path, "w") as f:
            await f.write(content)

    async def read_file(self, path: str) -> str:
        async with self.aiofiles.open(path, "r") as f:
            return await f.read()

    async def file_exists(self, path: str) -> bool:
        import os

        return os.path.exists(path)

    async def delete_file(self, path: str) -> None:
        import os

        if os.path.exists(path):
            os.remove(path)


class InMemoryRateLimiter:
    """In-memory rate limiter implementation."""

    def __init__(self, requests_per_window: int = 60, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
        import time
        from typing import Callable

        self._time: Callable[[], float] = time.time

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = self._time()
        window_start = now - self.window_seconds

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Clean old requests
        self._requests[client_ip] = [
            req_time for req_time in self._requests[client_ip] if req_time > window_start
        ]

        # Check if under limit
        if len(self._requests[client_ip]) < self.requests_per_window:
            self._requests[client_ip].append(now)
            return True

        return False

    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests in current window."""
        now = self._time()
        window_start = now - self.window_seconds

        if client_ip not in self._requests:
            return self.requests_per_window

        # Clean old requests
        self._requests[client_ip] = [
            req_time for req_time in self._requests[client_ip] if req_time > window_start
        ]

        return max(0, self.requests_per_window - len(self._requests[client_ip]))


# Factory functions


def create_production_dependencies(config: AppConfig) -> Dependencies:
    """Create dependencies for production use."""
    return Dependencies(
        config=config,
        http_client=HTTPXClient(timeout=30.0),
        cache=InMemoryCache(),
        filesystem=AIOFilesFileSystem(),
        rate_limiter=InMemoryRateLimiter(
            requests_per_window=config.rate_limit_requests,
            window_seconds=config.rate_limit_window,
        ),
    )


def create_test_dependencies(config: AppConfig) -> Dependencies:
    """Create dependencies for testing."""
    return Dependencies(
        config=config,
        http_client=HTTPXClient(timeout=30.0),  # Longer timeout for integration tests
        cache=InMemoryCache(),
        filesystem=AIOFilesFileSystem(),
        rate_limiter=InMemoryRateLimiter(
            requests_per_window=1000, window_seconds=60  # Higher limit for tests
        ),
    )


# Global dependency instance (for backward compatibility)
_default_deps: Optional[Dependencies] = None


def get_dependencies() -> Dependencies:
    """Get application dependencies."""
    global _default_deps
    if _default_deps is None:
        from .config import config

        _default_deps = create_production_dependencies(config)
    return _default_deps


def set_dependencies(deps: Dependencies) -> None:
    """Set application dependencies (for testing)."""
    global _default_deps
    _default_deps = deps
