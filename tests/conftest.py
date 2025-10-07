"""
Shared test fixtures and configuration for RunPod LLM Manager tests
"""

import asyncio
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from runpod_llm_manager.config import AppConfig
from runpod_llm_manager.dependencies import Dependencies, create_test_dependencies
from runpod_llm_manager.proxy_fastapi import app
from test_mocks import (
    create_completion_response,
    create_mock_dependencies,
    create_pod_create_response,
    create_pod_status_response,
    create_test_completion_request,
    create_test_config,
    create_test_pod_config,
)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Sync client for testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_config() -> AppConfig:
    """Test configuration with sensible defaults."""
    return create_test_config()


@pytest.fixture
def mock_deps(test_config: AppConfig) -> Dependencies:
    """Mock dependencies for testing."""
    return create_mock_dependencies(config=test_config)


@pytest.fixture
def mock_http_client():
    """Mock HTTP client with common responses."""
    from test_mocks import MockHTTPClient

    responses = {
        "http://mock-runpod:4010/v1/chat/completions": create_completion_response(),
        "http://mock-runpod:4010/graphql": create_pod_create_response(),
    }

    return MockHTTPClient(responses)


@pytest.fixture
def sample_completion_request() -> Dict[str, Any]:
    """Sample completion request for testing."""
    return create_test_completion_request()


@pytest.fixture
def sample_pod_config() -> Dict[str, Any]:
    """Sample pod configuration for testing."""
    return create_test_pod_config()


# Response fixtures
@pytest.fixture
def pod_create_response() -> Dict[str, Any]:
    """Mock pod creation response."""
    return create_pod_create_response()


@pytest.fixture
def pod_status_response() -> Dict[str, Any]:
    """Mock pod status response."""
    return create_pod_status_response()


@pytest.fixture
def completion_response() -> Dict[str, Any]:
    """Mock completion response."""
    return create_completion_response()


# Test data fixtures
@pytest.fixture
def valid_completion_payload() -> Dict[str, Any]:
    """Valid completion request payload."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 100,
        "temperature": 0.7,
    }


@pytest.fixture
def invalid_completion_payload() -> Dict[str, Any]:
    """Invalid completion request payload for testing validation."""
    return {
        "model": "",  # Invalid: empty model
        "messages": [],  # Invalid: empty messages
        "max_tokens": 5000,  # Invalid: too many tokens
        "temperature": 3.0,  # Invalid: temperature too high
    }


@pytest.fixture
def oversized_payload() -> Dict[str, Any]:
    """Payload that exceeds size limits."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "x" * 60000}],  # 60KB content
        "max_tokens": 100,
    }


# Rate limiting fixtures
@pytest.fixture
def rate_limited_client():
    """Client that has exceeded rate limits."""
    from test_mocks import MockRateLimiter

    limiter = MockRateLimiter(always_allow=False, remaining_requests=0)
    return limiter


@pytest.fixture
def healthy_rate_limiter():
    """Rate limiter with available requests."""
    from test_mocks import MockRateLimiter

    return MockRateLimiter(always_allow=True, remaining_requests=50)


# Cache fixtures
@pytest.fixture
def populated_cache():
    """Cache with some test data."""
    from test_mocks import MockCache

    cache = MockCache()
    cache.store["test-key"] = {"cached": "response"}
    return cache


@pytest.fixture
def empty_cache():
    """Empty cache for testing."""
    from test_mocks import MockCache

    return MockCache()
