"""
Functional tests for Docker image building and pod management with real APIs.
Tests the complete pipeline from custom image build to LLM inference.
"""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


@pytest.mark.functional
class TestDockerImagePodIntegrationReal:
    """Real integration tests for Docker image building and pod functionality."""

    @pytest.fixture(autouse=True)
    def check_environment(self):
        """Check that all required environment variables are set."""
        required_vars = ["DOCKER_USERNAME", "DOCKER_TOKEN", "RUNPOD_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Check Docker is available
        try:
            subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Docker is not available")

    @pytest.fixture
    def created_resources(self):
        """Track created resources for cleanup."""
        resources = {"pods": [], "images": []}
        yield resources

        # Cleanup
        for pod_id in resources["pods"]:
            try:
                # Note: Would need test_services fixture to terminate
                pass
            except:
                pass

    @pytest.mark.asyncio
    async def test_login_to_docker_registry(self, test_services):
        """Test logging into Docker registry."""
        # This is tested implicitly in build_and_push_image
        pass

    @pytest.mark.asyncio
    async def test_build_and_push_image(self, test_services, created_resources):
        """Test building and pushing a real Docker image."""
        model_name = "facebook/opt-125m"
        backend = "vllm"

        full_image_name = f"{test_services['docker_builder'].registry}/{test_services['docker_builder'].repository}:{test_services['docker_builder']._generate_image_tag(model_name, backend)}"

        # Create temporary directory for Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            dockerfile_content = test_services["docker_builder"]._generate_vllm_dockerfile(
                model_name
            )

            # Write Dockerfile
            dockerfile_path.write_text(dockerfile_content)

            # Build the image
            try:
                subprocess.run(
                    ["docker", "build", "-t", full_image_name, temp_dir],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                pytest.fail(f"Failed to build Docker image: {e.stderr}")

            # Push the image
            try:
                subprocess.run(
                    ["docker", "push", full_image_name], capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                pytest.fail(f"Failed to push Docker image: {e.stderr}")

        created_resources["images"].append(full_image_name)

    @pytest.mark.asyncio
    async def test_create_and_test_pod(self, test_services, created_resources):
        """Test creating a pod with custom image and testing LLM functionality."""
        model_name = "facebook/opt-125m"
        backend = "vllm"

        # First build and push image
        image_name = f"{test_services['docker_builder'].registry}/{test_services['docker_builder'].repository}:{test_services['docker_builder']._generate_image_tag(model_name, backend)}"

        # Build image (simplified - in real test would call build_and_push_image)
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            dockerfile_content = test_services["docker_builder"]._generate_vllm_dockerfile(
                model_name
            )
            dockerfile_path.write_text(dockerfile_content)

            subprocess.run(["docker", "build", "-t", image_name, temp_dir], check=True)
            subprocess.run(["docker", "push", image_name], check=True)

        created_resources["images"].append(image_name)

        pod_config = {
            "use_custom_image": True,
            "model_name": model_name,
            "gpu_type_id": "NVIDIA GeForce RTX 3090",
            "gpu_count": 1,
            "container_disk_gb": 20,
            "volume_gb": 0,
            "start_jupyter": False,
            "name": f"test-custom-image-pod-{int(time.time())}",
        }

        try:
            pod_id = await test_services["pod_service"].create_pod(pod_config)
            created_resources["pods"].append(pod_id)

            # Wait for pod to be ready
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    pod_status = await test_services["pod_service"].get_pod_status(pod_id)
                    status = pod_status.get("status", "UNKNOWN")

                    if status == "RUNNING":
                        break
                    elif status in ["FAILED", "TERMINATED"]:
                        pytest.fail(f"Pod failed with status: {status}")

                    await asyncio.sleep(10)

                except Exception as e:
                    await asyncio.sleep(10)

            else:
                pytest.fail(f"Pod {pod_id} did not become ready within {max_attempts * 10} seconds")

            # Test LLM functionality
            from runpod_llm_manager.proxy_fastapi_models import ChatCompletionRequest, ChatMessage

            request = ChatCompletionRequest(
                model=model_name,
                messages=[
                    ChatMessage(role="user", content="Hello! Please respond with a short greeting.")
                ],
                max_tokens=50,
                temperature=0.7,
                stream=False,
            )

            response = await test_services["llm_service"].process_completion_request(
                request, mode="pod", pod_id=pod_id
            )

            # Verify response structure
            assert "choices" in response
            assert len(response["choices"]) > 0
            assert "message" in response["choices"][0]
            assert "content" in response["choices"][0]["message"]

        except Exception as e:
            error_msg = str(e)
            if (
                "no longer any instances available" in error_msg
                or "instances available" in error_msg
            ):
                pytest.skip(f"Resource availability issue: {error_msg}")
            else:
                raise
        finally:
            # Cleanup pods
            for pod_id in created_resources["pods"]:
                try:
                    await test_services["pod_service"].terminate_pod(pod_id)
                except:
                    pass

    @pytest.mark.asyncio
    async def test_image_reuse_across_pods(self, test_services):
        """Test that the same image is reused for multiple pods with same config."""
        model_name = "facebook/opt-125m"
        backend = "vllm"

        # Build image first time
        image1 = await test_services["docker_builder"].get_or_build_image(model_name, backend)

        # Build image second time (should reuse)
        image2 = await test_services["docker_builder"].get_or_build_image(model_name, backend)

        # Should be the same image
        assert image1 == image2

    @pytest.mark.asyncio
    async def test_different_models_different_images(self, test_services):
        """Test that different models generate different images."""
        model1 = "facebook/opt-125m"
        model2 = "microsoft/DialoGPT-small"
        backend = "vllm"

        image1 = await test_services["docker_builder"].get_or_build_image(model1, backend)
        image2 = await test_services["docker_builder"].get_or_build_image(model2, backend)

        # Should be different images
        assert image1 != image2
        assert model1.replace("/", "-") in image1
        assert model2.replace("/", "-") in image2
