"""
Tests for business logic services
"""

import pytest
from unittest.mock import AsyncMock, patch

from services import LLMService, PodManagementService, HealthService
from test_mocks import create_test_config, create_mock_dependencies
from proxy_fastapi_models import ChatCompletionRequest, ChatMessage


class TestLLMService:
    """Test cases for LLM service."""

    @pytest.fixture
    def service(self, mock_deps):
        """Create LLM service with mock dependencies."""
        return LLMService(mock_deps)

    @pytest.fixture
    def sample_request(self):
        """Create sample completion request."""
        return ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="user", content="Hello, world!")
            ],
            max_tokens=100,
            temperature=0.7,
            stream=False
        )

    @pytest.mark.asyncio
    async def test_process_completion_request_cache_hit(self, service, sample_request, mock_deps):
        """Test cache hit scenario."""
        # Pre-populate cache
        cache_key = service._generate_cache_key(sample_request)
        cached_response = {"cached": "response"}
        await mock_deps.cache.set(cache_key, cached_response)

        # Process request
        response = await service.process_completion_request(sample_request)

        # Verify cache hit
        assert response == cached_response

        # Verify no HTTP call was made
        assert len(mock_deps.http_client.requests_made) == 0

    @pytest.mark.asyncio
    async def test_process_completion_request_cache_miss(self, service, sample_request, mock_deps):
        """Test cache miss scenario."""
        # Setup mock response
        mock_response = {"result": "from_api"}
        mock_deps.http_client.responses = {
            mock_deps.config.runpod_endpoint: mock_response
        }

        # Process request
        response = await service.process_completion_request(sample_request)

        # Verify API was called
        assert len(mock_deps.http_client.requests_made) == 1
        assert response == mock_response

        # Verify result was cached
        cache_key = service._generate_cache_key(sample_request)
        cached = await mock_deps.cache.get(cache_key)
        assert cached == mock_response

    def test_generate_cache_key_deterministic(self, service):
        """Test that cache key generation is deterministic."""
        request1 = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        request2 = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False
        )

        key1 = service._generate_cache_key(request1)
        key2 = service._generate_cache_key(request2)

        assert key1 == key2

    def test_generate_cache_key_unique(self, service):
        """Test that different requests generate different keys."""
        request1 = ChatCompletionRequest(
            model="test1",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        request2 = ChatCompletionRequest(
            model="test2",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False
        )

        key1 = service._generate_cache_key(request1)
        key2 = service._generate_cache_key(request2)

        assert key1 != key2


class TestPodManagementService:
    """Test cases for pod management service."""

    @pytest.fixture
    def service(self, mock_deps):
        """Create pod management service with mock dependencies."""
        return PodManagementService(mock_deps)

    @pytest.mark.asyncio
    async def test_create_pod_success(self, service, mock_deps, sample_pod_config):
        """Test successful pod creation."""
        # Setup mock response
        expected_pod_id = "test-pod-123"
        mock_deps.http_client.responses = {
            f"{mock_deps.config.runpod_endpoint.replace('/v1/chat/completions', '/graphql')}": {
                "data": {"podCreate": {"id": expected_pod_id}}
            }
        }

        # Create pod
        pod_id = await service.create_pod(sample_pod_config)

        # Verify result
        assert pod_id == expected_pod_id

        # Verify HTTP call was made
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert "mutation" in request["json"]["query"]
        assert "StartPod" in request["json"]["query"]

    @pytest.mark.asyncio
    async def test_get_pod_status_success(self, service, mock_deps):
        """Test successful pod status retrieval."""
        pod_id = "test-pod-123"
        mock_status = {
            "id": pod_id,
            "status": "RUNNING",
            "ip": "192.168.1.100"
        }

        mock_deps.http_client.responses = {
            f"{mock_deps.config.runpod_endpoint.replace('/v1/chat/completions', '/graphql')}": {
                "data": {"pod": mock_status}
            }
        }

        # Get pod status
        status = await service.get_pod_status(pod_id)

        # Verify result
        assert status == mock_status

        # Verify HTTP call
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert request["json"]["variables"]["id"] == pod_id


class TestHealthService:
    """Test cases for health service."""

    @pytest.fixture
    def service(self, mock_deps):
        """Create health service with mock dependencies."""
        return HealthService(mock_deps)

    @pytest.mark.asyncio
    async def test_get_health_status(self, service, mock_deps):
        """Test health status retrieval."""
        # The mock rate limiter returns 1000 by default (set in create_mock_dependencies)

        # Get health status
        health = await service.get_health_status("192.168.1.100")

        # Verify structure
        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert health["cache_dir"] == mock_deps.config.cache_dir
        assert health["endpoint"] == mock_deps.config.runpod_endpoint
        assert health["security"]["rate_limit_remaining"] == 1000  # Default from mock
        assert health["security"]["rate_limit_limit"] == mock_deps.config.rate_limit_requests

    @pytest.mark.asyncio
    async def test_get_health_status_rate_limited(self, service, mock_deps):
        """Test health status when rate limited."""
        # Create a new mock rate limiter with 0 remaining requests
        from test_mocks import MockRateLimiter
        exhausted_limiter = MockRateLimiter(always_allow=False, remaining_requests=0)
        mock_deps.rate_limiter = exhausted_limiter

        # Get health status
        health = await service.get_health_status("192.168.1.100")

        # Verify rate limit info
        assert health["security"]["rate_limit_remaining"] == 0