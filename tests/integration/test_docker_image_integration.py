"""
Integration tests for Docker image building, pod creation, and LLM functionality.
Tests the complete pipeline from custom image build to LLM inference.
"""

import pytest

from runpod_llm_manager.docker_builder import DockerImageBuilderService
from runpod_llm_manager.llm_service import LLMService
from runpod_llm_manager.pod_service import PodManagementService
from runpod_llm_manager.proxy_fastapi_models import ChatCompletionRequest, ChatMessage
from tests.fixtures.mock_services import create_mock_dependencies


@pytest.fixture(params=["mock", "real"])
def deps_type(request):
    """Parameter to choose between mock and real dependencies."""
    return request.param


@pytest.fixture
def test_deps(deps_type):
    """Dependencies for testing - mock or real."""
    if deps_type == "mock":
        return create_mock_dependencies()
    elif deps_type == "real":
        # Check if real environment is available
        import os

        if not os.getenv("RUNPOD_API_KEY"):
            pytest.skip("RUNPOD_API_KEY not set for real integration tests")
        if not os.getenv("DOCKER_USERNAME") or not os.getenv("DOCKER_TOKEN"):
            pytest.skip("Docker credentials not set for real integration tests")

        # Import real config and deps
        from runpod_llm_manager.config import AppConfig
        from runpod_llm_manager.dependencies import (
            AIOFilesFileSystem,
            Dependencies,
            HTTPXClient,
            InMemoryCache,
            InMemoryRateLimiter,
        )

        config = AppConfig(
            runpod_endpoint=os.getenv("RUNPOD_ENDPOINT", "https://api.runpod.ai"),
            runpod_api_key=os.getenv("RUNPOD_API_KEY"),
            cache_dir=os.getenv("CACHE_DIR", "./cache"),
            max_cache_size=int(os.getenv("MAX_CACHE_SIZE", "1000")),
            cache_size_bytes=int(os.getenv("CACHE_SIZE_BYTES", str(10 * 1024 * 1024 * 1024))),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            use_https=os.getenv("USE_HTTPS", "true").lower() == "true",
        )

        return Dependencies(
            config=config,
            http_client=HTTPXClient(timeout=60.0),
            cache=InMemoryCache(),
            rate_limiter=InMemoryRateLimiter(
                requests_per_window=config.rate_limit_requests,
                window_seconds=config.rate_limit_window,
            ),
            filesystem=AIOFilesFileSystem(),
        )


class TestDockerImagePodIntegration:
    """Integration tests for Docker image building and pod functionality."""

    @pytest.fixture
    def services(self, test_deps):
        """Create service instances with test dependencies."""
        llm_service = LLMService(test_deps)
        pod_service = PodManagementService(test_deps)
        llm_service.set_services(pod_service=pod_service)
        return {
            "docker_builder": DockerImageBuilderService(test_deps),
            "pod_service": pod_service,
            "llm_service": llm_service,
        }

    @pytest.mark.asyncio
    async def test_full_pipeline_custom_image_pod_llm(self, services, test_deps, deps_type):
        """Test complete pipeline: build image -> create pod -> run LLM inference."""
        model_name = "facebook/opt-125m"
        backend = "vllm"

        expected_pod_id = None
        mock_deps = None

        if deps_type == "mock":
            # Mock setup
            expected_pod_id = "test-custom-pod-123"
            mock_deps = test_deps
            mock_deps.http_client.responses = {
                f"https://index.docker.io/v2/llm-images/manifests/{services['docker_builder']._generate_image_tag(model_name, backend)}": {
                    "error": "Not found"
                },
                "https://api.runpod.ai/v2/pod": {"id": expected_pod_id},
                f"https://api.runpod.ai/v2/pod/{expected_pod_id}": {
                    "id": expected_pod_id,
                    "desiredStatus": "RUNNING",
                    "ip": "192.168.1.100",
                    "ports": "8888/http",
                },
                f"https://{expected_pod_id}-8000.proxy.runpod.net/v1/chat/completions": {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! This is a test response from the custom pod.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
                },
            }
        elif deps_type == "real":
            # Real test - skip expensive operations in CI
            pytest.skip("Real full pipeline test is expensive and should be run manually")

        # Step 1: Build or get custom Docker image
        image_name = await services["docker_builder"].get_or_build_image(model_name, backend)
        if deps_type == "mock":
            assert image_name.startswith(f"{services['docker_builder'].registry}/llm-images:")
            assert backend in image_name
            assert model_name.replace("/", "-").replace(":", "-") in image_name
        # For real, just check it's a string
        assert isinstance(image_name, str)

        # Step 2: Create pod with custom image
        pod_config = {
            "use_custom_image": True,
            "model_name": model_name,
            "gpu_type_id": "NVIDIA GeForce RTX 3090",
            "gpu_count": 1,
            "container_disk_gb": 20,
            "volume_gb": 0,
            "start_jupyter": False,
            "name": "test-custom-image-pod",
        }

        pod_id = await services["pod_service"].create_pod(pod_config)
        if deps_type == "mock":
            assert pod_id == expected_pod_id
        else:
            assert pod_id  # For real, just check it exists

        # Step 3: Check pod status
        pod_status = await services["pod_service"].get_pod_status(pod_id)
        assert pod_status["id"] == pod_id
        if deps_type == "mock":
            assert pod_status["status"] == "RUNNING"
            assert pod_status["ip"] == "192.168.1.100"
            assert len(pod_status["ports"]) > 0

        # Step 4: Test LLM completion on the custom pod
        request = ChatCompletionRequest(
            model=model_name,
            messages=[ChatMessage(role="user", content="Hello, test message!")],
            max_tokens=50,
            temperature=0.7,
            stream=False,
        )

        response = await services["llm_service"].process_completion_request(
            request, mode="pod", pod_id=pod_id
        )

        # Verify response structure
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]

        if deps_type == "mock":
            assert (
                "Hello! This is a test response from the custom pod."
                in response["choices"][0]["message"]["content"]
            )

        # Step 5: Clean up - terminate pod
        await services["pod_service"].terminate_pod(pod_id)

        if deps_type == "mock" and mock_deps:
            # Verify termination call was made
            assert (
                len(mock_deps.http_client.requests_made) >= 4
            )  # No registry check for custom registry
            delete_requests = [
                r for r in mock_deps.http_client.requests_made if r.get("method") == "DELETE"
            ]
            assert len(delete_requests) == 1
            assert f"api.runpod.ai/v2/pod/{pod_id}" in delete_requests[0]["url"]

    @pytest.mark.asyncio
    async def test_image_reuse_across_pods(self, services, test_deps, deps_type):
        """Test that the same image is reused for multiple pods with same config."""
        model_name = "facebook/opt-125m"
        backend = "vllm"

        if deps_type == "mock":
            # Mock registry - image exists
            tag = services["docker_builder"]._generate_image_tag(model_name, backend)
            registry = services["docker_builder"].registry
            test_deps.http_client.responses = {
                f"https://index.docker.io/v2/llm-images/manifests/{tag}": {"schemaVersion": 2},
                "https://api.runpod.ai/v2/pod": [{"id": "pod-1"}, {"id": "pod-2"}],
            }

        # Build image first time
        image1 = await services["docker_builder"].get_or_build_image(model_name, backend)

        # Build image second time (should reuse)
        image2 = await services["docker_builder"].get_or_build_image(model_name, backend)

        # Should be the same image
        assert image1 == image2

        if deps_type == "mock":
            # Verify only one registry check was made (image was cached as existing)
            registry_requests = [
                r
                for r in test_deps.http_client.requests_made
                if "index.docker.io" in r.get("url", "")
            ]
            # Note: With custom registry, we skip the check and assume it exists
            # So no registry requests are made, which is expected behavior
            assert len(registry_requests) == 0  # No checks made for custom registry
