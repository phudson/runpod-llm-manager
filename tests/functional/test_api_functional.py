"""
Functional tests for RunPod API integration.
Tests template fetching, pod creation, and LLM functionality with real APIs.
"""

import asyncio

import pytest

from runpod_llm_manager.pod_service import PodManagementService


@pytest.mark.functional
class TestAPITemplateFetching:
    """Test template fetching from RunPod API."""

    @pytest.mark.asyncio
    async def test_template_fetching_basic_auth(self, test_deps):
        """Test basic API authentication and template fetching."""
        try:
            # Test basic authentication with REST API first
            templates_response = await test_deps.http_client.get(
                f"{test_deps.config.runpod_endpoint}/v2/templates",
                headers={"Authorization": f"Bearer {test_deps.config.runpod_api_key}"},
            )

            # If we get a response (even if empty), authentication worked
            assert isinstance(templates_response, list) or isinstance(templates_response, dict)
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "Not Found" in error_msg:
                pytest.skip("Templates API endpoint not available")
            elif "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg:
                pytest.skip("API authentication failed")
            else:
                raise

    @pytest.mark.asyncio
    async def test_template_fetching_list_templates(self, test_deps):
        """Test fetching available templates."""
        try:
            templates_response = await test_deps.http_client.get(
                "https://api.runpod.ai/v2/templates",
                headers={"Authorization": f"Bearer {test_deps.config.runpod_api_key}"},
            )

            assert isinstance(templates_response, list)
            assert len(templates_response) > 0

            # Check for PyTorch templates
            pytorch_templates = [
                t
                for t in templates_response
                if "pytorch" in t.get("name", "").lower() or "torch" in t.get("name", "").lower()
            ]
            assert len(pytorch_templates) > 0
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "Not Found" in error_msg:
                pytest.skip("Templates API endpoint not available")
            elif "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg:
                pytest.skip("API authentication failed")
            else:
                raise

    @pytest.mark.asyncio
    async def test_template_id_resolution(self, test_services):
        """Test template ID lookup by name."""
        try:
            templates_response = await test_services["pod_service"].deps.http_client.get(
                "https://api.runpod.ai/v2/templates",
                headers={
                    "Authorization": f"Bearer {test_services['pod_service'].deps.config.runpod_api_key}"
                },
            )

            pytorch_templates = [
                t
                for t in templates_response
                if isinstance(t, dict)
                and ("pytorch" in t.get("name", "").lower() or "torch" in t.get("name", "").lower())
            ]

            if pytorch_templates:
                test_template_name = pytorch_templates[0].get("name")
                template_id = await test_services["pod_service"].get_template_id_by_name(
                    test_template_name
                )
                assert template_id
                assert isinstance(template_id, str)
            else:
                pytest.skip("No PyTorch templates available for testing")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "Not Found" in error_msg:
                pytest.skip("Templates API endpoint not available")
            elif "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg:
                pytest.skip("API authentication failed")
            else:
                raise


@pytest.mark.functional
class TestAPIPodCreation:
    """Test pod creation with real API."""

    @pytest.mark.asyncio
    async def test_pod_creation_with_template(self, test_services):
        """Test creating a pod with a real template."""
        try:
            # Get available templates
            templates_response = await test_services["pod_service"].deps.http_client.get(
                "https://api.runpod.ai/v2/templates",
                headers={
                    "Authorization": f"Bearer {test_services['pod_service'].deps.config.runpod_api_key}"
                },
            )

            pytorch_templates = [
                t
                for t in templates_response
                if isinstance(t, dict)
                and ("pytorch" in t.get("name", "").lower() or "torch" in t.get("name", "").lower())
            ]

            if not pytorch_templates:
                pytest.skip("No PyTorch templates available")

            template_name = pytorch_templates[0].get("name")

            pod_config = {
                "template_id": template_name,
                "gpu_type_id": "NVIDIA GeForce RTX 5090",
            }

            pod_id = await test_services["pod_service"].create_pod(pod_config)
            assert pod_id
            assert isinstance(pod_id, str)

            # Check status
            status = await test_services["pod_service"].get_pod_status(pod_id)
            assert status
            assert "id" in status

            # Clean up
            await test_services["pod_service"].terminate_pod(pod_id)

        except Exception as e:
            error_msg = str(e)
            # Allow expected failures due to resource availability
            if (
                "no longer any instances available" in error_msg
                or "instances available" in error_msg
            ):
                pytest.skip(f"Resource availability issue: {error_msg}")
            elif "Template not found" in error_msg:
                pytest.skip(f"Template issue: {error_msg}")
            elif "404" in error_msg or "Not Found" in error_msg:
                pytest.skip("Templates API endpoint not available")
            elif "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg:
                pytest.skip("API authentication failed")
            else:
                raise


@pytest.mark.functional
class TestAPIFullLLMIntegration:
    """Test complete LLM integration with custom images."""

    @pytest.mark.asyncio
    async def test_full_llm_integration_custom_image(self, test_services):
        """Test complete LLM pipeline with custom vLLM image."""
        model_name = "facebook/opt-125m"

        pod_config = {
            "template_id": "debian",
            "gpu_type_id": "NVIDIA RTX A4000",
            "use_custom_image": True,
            "model_name": model_name,
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 512,
            "enforce_eager": True,
            "name": "test-llm-pod",
        }

        pod_id = None
        try:
            pod_id = await test_services["pod_service"].create_pod(pod_config)
            assert pod_id

            # Wait for pod to be ready
            max_wait_time = 600
            poll_interval = 10
            start_time = asyncio.get_event_loop().time()

            while (asyncio.get_event_loop().time() - start_time) < max_wait_time:
                status = await test_services["pod_service"].get_pod_status(pod_id)
                pod_status = status.get("status", "UNKNOWN")

                if pod_status == "RUNNING":
                    break
                elif pod_status in ["TERMINATED", "FAILED", "ERROR"]:
                    raise Exception(f"Pod failed with status: {pod_status}")

                await asyncio.sleep(poll_interval)
            else:
                raise Exception(f"Pod did not reach RUNNING status within {max_wait_time} seconds")

            # Test LLM inference
            pod_details = await test_services["pod_service"].get_pod_status(pod_id)
            ports = pod_details.get("ports", [])

            assert ports, "No ports available on pod"

            # Find vLLM port
            vllm_port = None
            for port_info in ports:
                if isinstance(port_info, dict) and port_info.get("privatePort") == 8000:
                    vllm_port = 8000
                    break

            if not vllm_port:
                for port_info in ports:
                    if isinstance(port_info, dict) and port_info.get("type") == "http":
                        vllm_port = port_info.get("privatePort")
                        break

            assert vllm_port, f"No suitable port found. Available ports: {ports}"

            endpoint_url = f"https://{pod_id}-{vllm_port}.proxy.runpod.net"

            # Test chat completion
            test_payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Hello, can you tell me what 2+2 equals?"}
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            }

            completion_response = await test_services["pod_service"].deps.http_client.post(
                f"{endpoint_url}/v1/chat/completions", json=test_payload
            )

            assert "choices" in completion_response
            assert len(completion_response["choices"]) > 0
            response_text = completion_response["choices"][0].get("message", {}).get("content", "")
            assert response_text

            # Clean up
            await test_services["pod_service"].terminate_pod(pod_id)

        except Exception as e:
            error_msg = str(e)
            if pod_id:
                try:
                    await test_services["pod_service"].terminate_pod(pod_id)
                except:
                    pass

            if (
                "no longer any instances available" in error_msg
                or "instances available" in error_msg
            ):
                pytest.skip(f"Resource availability issue: {error_msg}")
            elif "Template not found" in error_msg:
                pytest.skip(f"Template issue: {error_msg}")
            else:
                raise
