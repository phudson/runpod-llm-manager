"""
Mock implementations for testing RunPod LLM Manager
Provides testable versions of all dependencies
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from runpod_llm_manager.config import AppConfig
from runpod_llm_manager.dependencies import (
    CacheProtocol,
    Dependencies,
    FileSystemProtocol,
    HTTPClientProtocol,
    RateLimiterProtocol,
)


class MockHTTPClient:
    """Mock HTTP client for testing."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.requests_made: List[Dict[str, Any]] = []
        self.should_fail = False
        self.fail_with_exception = None
        self.response_index = {}  # Track response index per URL

    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Record request and return mock response."""
        self.requests_made.append({"method": "POST", "url": url, "json": json, "headers": headers})

        if self.should_fail:
            if self.fail_with_exception:
                raise self.fail_with_exception
            raise Exception("Mock HTTP client configured to fail")

        url_responses = self.responses.get(url, {"error": "Mock not configured for this URL"})

        # Handle multiple responses for the same URL (return them in sequence)
        if isinstance(url_responses, list):
            index = self.response_index.get(url, 0)
            response = url_responses[index] if index < len(url_responses) else url_responses[-1]
            self.response_index[url] = index + 1
            return response
        else:
            return url_responses

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock GET request."""
        self.requests_made.append({"method": "GET", "url": url})

        if self.should_fail:
            if self.fail_with_exception:
                raise self.fail_with_exception
            raise Exception("Mock HTTP client configured to fail")

        return self.responses.get(url, {"error": "Mock not configured for this URL"})

    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock DELETE request."""
        self.requests_made.append({"method": "DELETE", "url": url})

        if self.should_fail:
            if self.fail_with_exception:
                raise self.fail_with_exception
            raise Exception("Mock HTTP client configured to fail")

        return self.responses.get(url, {})  # DELETE usually returns empty dict on success

    async def post_stream(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Mock streaming POST request."""
        self.requests_made.append(
            {"method": "POST_STREAM", "url": url, "json": json, "headers": headers}
        )

        if self.should_fail:
            if self.fail_with_exception:
                raise self.fail_with_exception
            raise Exception("Mock HTTP client configured to fail")

        # Return async generator that yields mock data
        async def mock_generator():
            yield 'data: {"chunk": "test"}\n\n'
            yield "data: [DONE]\n\n"

        return mock_generator()


class MockCache:
    """Mock cache for testing."""

    def __init__(self, initial_data: Optional[Dict[str, Dict[str, Any]]] = None):
        self.store = initial_data or {}
        self.operations: List[Dict[str, Any]] = []
        self.access_times: Dict[str, float] = {}
        import time

        self._time = time.time

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        self.operations.append({"operation": "get", "key": key})
        if key in self.store:
            self.access_times[key] = self._time()
            return self.store[key]
        return None

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set cached value."""
        self.operations.append({"operation": "set", "key": key, "value": value})
        self.store[key] = value
        self.access_times[key] = self._time()

    async def delete(self, key: str) -> None:
        """Delete cached value."""
        self.operations.append({"operation": "delete", "key": key})
        self.store.pop(key, None)
        self.access_times.pop(key, None)

    async def clear(self) -> None:
        """Clear all cached values."""
        self.operations.append({"operation": "clear"})
        self.store.clear()
        self.access_times.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = self._time()
        total_size = sum(len(str(v).encode("utf-8")) for v in self.store.values())

        recent_accesses = sum(1 for t in self.access_times.values() if now - t < 3600)

        return {
            "entries": len(self.store),
            "total_size_bytes": total_size,
            "recent_accesses": recent_accesses,
            "oldest_entry_age": min((now - t for t in self.access_times.values()), default=0),
            "newest_entry_age": max((now - t for t in self.access_times.values()), default=0),
        }


class MockFileSystem:
    """Mock file system for testing."""

    def __init__(self, initial_files: Optional[Dict[str, str]] = None):
        self.files = initial_files or {}
        self.operations: List[Dict[str, Any]] = []

    async def write_file(self, path: str, content: str) -> None:
        """Write content to mock file."""
        self.operations.append({"operation": "write", "path": path, "content": content})
        self.files[path] = content

    async def read_file(self, path: str) -> str:
        """Read content from mock file."""
        self.operations.append({"operation": "read", "path": path})
        if path not in self.files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.files[path]

    async def file_exists(self, path: str) -> bool:
        """Check if mock file exists."""
        self.operations.append({"operation": "exists", "path": path})
        return path in self.files

    async def delete_file(self, path: str) -> None:
        """Delete mock file."""
        self.operations.append({"operation": "delete", "path": path})
        self.files.pop(path, None)


class MockRateLimiter:
    """Mock rate limiter for testing."""

    def __init__(
        self, always_allow: bool = True, remaining_requests: int = 100, limit_after: int = 10
    ):
        self.always_allow = always_allow
        self.remaining_requests_value = remaining_requests
        self.limit_after = limit_after  # Start limiting after this many requests
        self.request_count = 0
        self.calls: List[Dict[str, Any]] = []

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        self.calls.append({"method": "is_allowed", "client_ip": client_ip})
        self.request_count += 1

        if self.always_allow:
            return True

        # Start limiting after limit_after requests
        return self.request_count <= self.limit_after

    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests."""
        self.calls.append({"method": "get_remaining_requests", "client_ip": client_ip})

        if self.always_allow:
            return self.remaining_requests_value

        # When not always allowing, return the configured remaining_requests value
        return self.remaining_requests_value


# Test configuration factory
def create_test_config(**overrides) -> AppConfig:
    """Create test configuration with sensible defaults."""
    defaults = {
        "runpod_endpoint": "http://mock-runpod:4010/v1/chat/completions",
        "runpod_api_key": "test-api-key",
        "cache_dir": "/tmp/test_cache",
        "rate_limit_requests": 1000,  # High limit for tests
        "rate_limit_window": 60,
        "max_request_size": 1048576,
        "max_content_length": 50000,
        "enable_profiling": False,
        "use_https": False,
        "test_mode": True,
        "mock_runpod_url": "http://mock-runpod:4010",
    }

    # Apply overrides
    config_dict = {**defaults, **overrides}

    # Create config object manually since we can't use the normal constructor
    config = AppConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# Mock dependency factories
def create_mock_dependencies(
    config: Optional[AppConfig] = None,
    http_responses: Optional[Dict[str, Any]] = None,
    cache_data: Optional[Dict[str, Dict[str, Any]]] = None,
    filesystem_data: Optional[Dict[str, str]] = None,
) -> Dependencies:
    """Create mock dependencies for testing."""
    if config is None:
        config = create_test_config()

    return Dependencies(
        config=config,
        http_client=MockHTTPClient(http_responses),
        cache=MockCache(cache_data),
        filesystem=MockFileSystem(filesystem_data),
        rate_limiter=MockRateLimiter(always_allow=True, remaining_requests=1000),
    )


# Common test fixtures
def create_pod_create_response(pod_id: str = "test-pod-123") -> Dict[str, Any]:
    """Create mock pod creation response."""
    return {"data": {"podFindAndDeployOnDemand": {"id": pod_id}}}


def create_template_lookup_response() -> Dict[str, Any]:
    """Create mock template lookup response matching real API."""
    return {
        "data": {
            "myself": {
                "podTemplates": [
                    {
                        "id": "runpod-torch-v21",
                        "name": "Runpod Pytorch 2.1",
                        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                    },
                    {
                        "id": "runpod-torch-v220",
                        "name": "Runpod Pytorch 2.2.0",
                        "imageName": "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
                    },
                    {
                        "id": "runpod-torch-v240",
                        "name": "Runpod Pytorch 2.4.0",
                        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                    },
                    {
                        "id": "runpod-torch-v280",
                        "name": "Runpod Pytorch 2.8.0",
                        "imageName": "runpod/pytorch:1.0.1-cu1281-torch280-ubuntu2404",
                    },
                    {
                        "id": "runpod-vscode",
                        "name": "Runpod VS Code Server",
                        "imageName": "runpod/vscode-server:0.0.0",
                    },
                    {"id": "cw3nka7d08", "name": "ComfyUI", "imageName": "runpod/comfyui:latest"},
                    {
                        "id": "runpod-torch-v240-rocm61",
                        "name": "Runpod Pytorch 2.4.0 ROCm 6.1",
                        "imageName": "runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04",
                    },
                    {
                        "id": "runpod-ubuntu",
                        "name": "Runpod Ubuntu 20.04",
                        "imageName": "runpod/base:0.7.0-ubuntu2004",
                    },
                    {
                        "id": "runpod-ubuntu-2204",
                        "name": "Runpod Ubuntu 22.04",
                        "imageName": "runpod/base:1.0.1-ubuntu2204",
                    },
                    {
                        "id": "runpod-ubuntu-2404",
                        "name": "Runpod Ubuntu 24.04",
                        "imageName": "runpod/base:1.0.1-ubuntu2404",
                    },
                ]
            }
        }
    }


def create_pod_status_response(
    pod_id: str = "test-pod-123", status: str = "RUNNING", ip: str = "192.168.1.100"
) -> Dict[str, Any]:
    """Create mock pod status response."""
    return {
        "data": {
            "pod": {
                "id": pod_id,
                "desiredStatus": status,
                "ipAddress": {"address": ip},
                "ports": [{"ip": ip, "privatePort": 8000, "publicPort": 8000, "type": "http"}],
            }
        }
    }


def create_completion_response(content: str = "Test response") -> Dict[str, Any]:
    """Create mock completion response."""
    return {
        "id": "test-completion",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{"text": content, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def create_chat_completion_response(
    content: str = "Hello! This is a test response.",
) -> Dict[str, Any]:
    """Create mock OpenAI chat completion response."""
    # If no custom content provided, use a response that looks like code for coding assistant tests
    if content == "Hello! This is a test response.":
        content = """```python
def reverse_string(s):
    return s[::-1]

# Example usage:
result = reverse_string("hello")
print(result)  # Output: "olleh"
```"""
    """Create mock OpenAI chat completion response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
    }


# Test data helpers
def create_test_completion_request(
    model: str = "test-model",
    messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 100,
) -> Dict[str, Any]:
    """Create test completion request data."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello, world!"}]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }


def create_test_pod_config(
    model_store_id: str = "deepseek-ai/deepseek-coder-33b-awq",
    gpu_type: str = "NVIDIA GeForce RTX 5090",
    runtime_seconds: int = 1800,
) -> Dict[str, Any]:
    """Create test pod configuration."""
    return {
        "modelStoreId": model_store_id,
        "gpu_type_id": gpu_type,
        "runtime_seconds": runtime_seconds,
        "template_id": "Runpod Pytorch 2.8.0",
    }
