"""
Tests for business logic services
"""

from unittest.mock import AsyncMock, patch

import pytest

from runpod_llm_manager.docker_builder import DockerImageBuilderService
from runpod_llm_manager.health_service import HealthService
from runpod_llm_manager.llm_service import LLMService
from runpod_llm_manager.pod_service import PodManagementService
from runpod_llm_manager.proxy_fastapi_models import ChatCompletionRequest, ChatMessage
from runpod_llm_manager.serverless_service import ServerlessService
from tests.fixtures.mock_services import (
    MockRateLimiter,
    create_mock_dependencies,
    create_test_config,
)


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
            messages=[ChatMessage(role="user", content="Hello, world!")],
            max_tokens=100,
            temperature=0.7,
            stream=False,
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
        mock_deps.http_client.responses = {mock_deps.config.runpod_endpoint: mock_response}

        # Set endpoint URL for the service (required for cache miss)
        service.set_endpoint_url(mock_deps.config.runpod_endpoint)

        # Process request (pass endpoint_url explicitly)
        response = await service.process_completion_request(
            sample_request, mock_deps.config.runpod_endpoint
        )

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
            stream=False,
        )
        request2 = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False,
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
            stream=False,
        )
        request2 = ChatCompletionRequest(
            model="test2",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False,
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
        # Setup mock responses for template lookup and pod creation
        expected_pod_id = "test-pod-123"
        template_id = "runpod-torch-v280"  # Correct ID for "Runpod Pytorch 2.8.0"

        # Mock REST API response for template lookup
        rest_templates_response = [
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
                "imageName": "runpod/pytorch:2.4.0-py3.10-rocm6.0-ubuntu22.04",
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

        # Mock REST API response for pod creation
        pod_creation_response = {"id": expected_pod_id}

        mock_deps.http_client.responses = {
            "https://api.runpod.ai/v2/templates": rest_templates_response,
            "https://api.runpod.ai/v2/pod": pod_creation_response,
        }

        # Create pod
        pod_id = await service.create_pod(sample_pod_config)

        # Verify result
        assert pod_id == expected_pod_id

        # Verify HTTP calls were made (template lookup + pod creation)
        assert len(mock_deps.http_client.requests_made) == 2

        # Check template lookup call (REST API)
        template_request = mock_deps.http_client.requests_made[0]
        assert template_request["method"] == "GET"
        assert "api.runpod.ai/v2/templates" in template_request["url"]

        # Check pod creation call (REST API)
        pod_request = mock_deps.http_client.requests_made[1]
        assert pod_request["method"] == "POST"
        assert "api.runpod.ai/v2/pod" in pod_request["url"]
        assert pod_request["json"]["templateId"] == template_id
        # Check that environment variables are included
        assert "env" in pod_request["json"]
        env_vars = pod_request["json"]["env"]
        assert isinstance(env_vars, list)

    @pytest.mark.asyncio
    async def test_get_pod_status_success(self, service, mock_deps):
        """Test successful pod status retrieval."""
        pod_id = "test-pod-123"
        mock_status = {
            "id": pod_id,
            "status": "RUNNING",
            "ip": "192.168.1.100",
            "ports": [{"privatePort": 8888, "type": "http"}, {"privatePort": 22, "type": "tcp"}],
        }

        mock_deps.http_client.responses = {
            f"https://api.runpod.ai/v2/pod/{pod_id}": {
                "id": pod_id,
                "desiredStatus": "RUNNING",
                "ip": "192.168.1.100",
                "ports": "8888/http,22/tcp",
            }
        }

        # Get pod status
        status = await service.get_pod_status(pod_id)

        # Verify result
        assert status == mock_status

        # Verify HTTP call (REST API)
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert request["method"] == "GET"
        assert f"api.runpod.ai/v2/pod/{pod_id}" in request["url"]

    @pytest.mark.asyncio
    async def test_terminate_pod_success(self, service, mock_deps):
        """Test successful pod termination."""
        pod_id = "test-pod-123"

        # Setup mock response (DELETE returns empty dict on success)
        mock_deps.http_client.responses = {f"https://api.runpod.ai/v2/pod/{pod_id}": {}}

        # Terminate pod
        await service.terminate_pod(pod_id)

        # Verify HTTP call was made
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert request["method"] == "DELETE"
        assert f"api.runpod.ai/v2/pod/{pod_id}" in request["url"]


class TestServerlessService:
    """Test cases for serverless service."""

    @pytest.fixture
    def service(self, mock_deps):
        """Create serverless service with mock dependencies."""
        return ServerlessService(mock_deps)

    @pytest.mark.asyncio
    async def test_create_vllm_endpoint_success(self, service, mock_deps):
        """Test successful vLLM endpoint creation."""
        expected_endpoint_id = "endpoint-123"

        mock_deps.http_client.responses = {
            "https://api.runpod.ai/v2/endpoint": {"id": expected_endpoint_id}
        }

        config = {
            "template_id": "vllm-template",
            "gpu_type_id": "NVIDIA GeForce RTX 5090",
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "gpu_count": 1,
            "worker_count": 1,
        }

        endpoint_id = await service.create_vllm_endpoint(config)

        assert endpoint_id == expected_endpoint_id

        # Verify HTTP call (REST API)
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert request["method"] == "POST"
        assert "api.runpod.ai/v2/endpoint" in request["url"]

        # Verify vLLM environment variables are set
        env_vars = request["json"]["env"]
        env_keys = [env["key"] for env in env_vars]
        assert "VLLM_MODEL" in env_keys
        assert "MODEL_NAME" in env_keys
        assert "RUNPOD_USE_MODEL_STORE" in env_keys

    @pytest.mark.asyncio
    async def test_get_endpoint_status_success(self, service, mock_deps):
        """Test successful endpoint status retrieval."""
        endpoint_id = "endpoint-123"
        mock_status = {
            "id": endpoint_id,
            "name": "vllm-endpoint",
            "type": "serverless",
            "gpuCount": 1,
            "workerCount": 1,
            "workers": [{"id": "worker-1", "status": "RUNNING"}],
        }

        mock_deps.http_client.responses = {
            f"https://api.runpod.ai/v2/endpoint/{endpoint_id}": mock_status
        }

        status = await service.get_endpoint_status(endpoint_id)

        assert status == mock_status

        # Verify HTTP call
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert request["method"] == "GET"
        assert f"api.runpod.ai/v2/endpoint/{endpoint_id}" in request["url"]

    @pytest.mark.asyncio
    async def test_delete_endpoint_success(self, service, mock_deps):
        """Test successful endpoint deletion."""
        endpoint_id = "endpoint-123"

        mock_deps.http_client.responses = {f"https://api.runpod.ai/v2/endpoint/{endpoint_id}": {}}

        await service.delete_endpoint(endpoint_id)

        # Verify HTTP call
        assert len(mock_deps.http_client.requests_made) == 1
        request = mock_deps.http_client.requests_made[0]
        assert request["method"] == "DELETE"
        assert f"api.runpod.ai/v2/endpoint/{endpoint_id}" in request["url"]

    @pytest.mark.asyncio
    async def test_get_model_store_status(self, service, mock_deps):
        """Test model store status check."""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        # Mock the model store API response
        mock_response = {
            "model_name": model_name,
            "cached": True,
            "cache_location": "/runpod-volume/model-cache",
            "size_gb": 15.2,
            "last_accessed": "2024-01-01T00:00:00Z",
        }
        mock_deps.http_client.responses = {
            f"{mock_deps.config.runpod_endpoint}/model-store/status/{model_name}": mock_response
        }

        status = await service.get_model_store_status(model_name)

        assert status["model_name"] == model_name
        assert status["cached"] is True
        assert "cache_location" in status
        assert "size_gb" in status


class TestLLMServiceModes:
    """Test cases for LLM service with different execution modes."""

    @pytest.fixture
    def service(self, mock_deps):
        """Create LLM service with mock dependencies."""
        return LLMService(mock_deps)

    @pytest.mark.asyncio
    async def test_process_completion_request_serverless_mode(
        self, service, sample_completion_request, mock_deps
    ):
        """Test LLM service in serverless mode."""
        endpoint_id = "endpoint-123"
        mock_response = {"choices": [{"message": {"content": "Hello from serverless!"}}]}

        mock_deps.http_client.responses = {
            f"https://{endpoint_id}.runpod.net/v1/chat/completions": mock_response
        }

        response = await service.process_completion_request(
            sample_completion_request, f"https://{endpoint_id}.runpod.net/v1/chat/completions"
        )

        assert response == mock_response

    @pytest.mark.asyncio
    async def test_process_completion_request_pod_mode(
        self, service, sample_completion_request, mock_deps
    ):
        """Test LLM service in pod mode."""
        pod_id = "pod-123"
        mock_response = {"choices": [{"message": {"content": "Hello from pod!"}}]}

        # Mock pod status response
        mock_deps.http_client.responses = {
            f"https://api.runpod.ai/v2/pod/{pod_id}": {
                "id": pod_id,
                "desiredStatus": "RUNNING",
                "ports": "8888/http",
            },
            f"https://{pod_id}-8888.proxy.runpod.net/v1/chat/completions": mock_response,
        }

        response = await service.process_completion_request(
            sample_completion_request, f"https://{pod_id}-8888.proxy.runpod.net/v1/chat/completions"
        )

        assert response == mock_response


class TestDockerImageBuilderService:
    """Test cases for Docker image builder service."""

    @pytest.fixture
    def service(self, mock_deps):
        """Create Docker image builder service with mock dependencies."""
        return DockerImageBuilderService(mock_deps)

    @pytest.mark.asyncio
    async def test_get_or_build_image_registry_hit(self, service, mock_deps):
        """Test registry hit scenario."""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        backend = "vllm"
        expected_image = (
            f"{service.registry}/llm-images:{service._generate_image_tag(model_name, backend)}"
        )

        # Mock registry API response indicating image exists
        mock_deps.http_client.responses = {
            f"https://index.docker.io/v2/llm-images/manifests/{service._generate_image_tag(model_name, backend)}": {
                "schemaVersion": 2
            }
        }

        # Get image
        image_name = await service.get_or_build_image(model_name, backend)

        # Verify result
        assert image_name == expected_image

        # Verify no registry check was made (custom registry assumed to exist)
        assert len(mock_deps.http_client.requests_made) == 0

    @pytest.mark.asyncio
    async def test_get_or_build_image_registry_miss(self, service, mock_deps):
        """Test registry miss scenario."""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        backend = "vllm"
        expected_image = (
            f"{service.registry}/llm-images:{service._generate_image_tag(model_name, backend)}"
        )

        # Mock registry API response indicating image doesn't exist
        mock_deps.http_client.responses = {
            f"https://index.docker.io/v2/llm-images/manifests/{service._generate_image_tag(model_name, backend)}": {
                "error": "Not found"
            }
        }

        # Get image (should build since it doesn't exist)
        image_name = await service.get_or_build_image(model_name, backend)

        # Verify result
        assert image_name == expected_image

        # Verify no registry check was made (custom registry assumed to exist)
        assert len(mock_deps.http_client.requests_made) == 0

    def test_generate_image_tag_deterministic(self, service):
        """Test that image tag generation is deterministic."""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        backend = "vllm"

        tag1 = service._generate_image_tag(model_name, backend)
        tag2 = service._generate_image_tag(model_name, backend)

        assert tag1 == tag2
        assert tag1.startswith("vllm-")
        assert "mistralai-mistral-7b-instruct-v0.2" in tag1


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
        exhausted_limiter = MockRateLimiter(always_allow=False, remaining_requests=0)
        mock_deps.rate_limiter = exhausted_limiter

        # Get health status
        health = await service.get_health_status("192.168.1.100")

        # Verify rate limit info
        assert health["security"]["rate_limit_remaining"] == 0
