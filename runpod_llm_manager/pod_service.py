"""
Pod Service
Handles RunPod pod management operations with intelligent hybrid approach.
"""

import logging
from typing import Any, Dict

from .dependencies import Dependencies
from .docker_builder import DockerImageBuilderService

logger = logging.getLogger(__name__)


class PodManagementService:
    """Service for RunPod pod management operations with intelligent hybrid approach.

    Uses templates for speed/cost optimization, custom images for maximum compatibility:
    - Popular models/backends: Try templates first (faster, cheaper)
    - Custom/unsupported models: Custom images (guaranteed compatibility)
    - Explicit custom_image=True: Custom images only (strict delivery guarantee)
    """

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.docker_builder = DockerImageBuilderService(deps)

    def is_model_supported_by_templates(self, model_name: str, backend: str = "vllm") -> bool:
        """Check if a model/backend combination is supported by RunPod templates.

        This is a heuristic - templates generally support popular models with standard backends.
        """
        if backend != "vllm":
            return False  # Templates likely only support vLLM

        # Common models that are typically supported by vLLM templates
        supported_models = {
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "distilgpt2",
        }

        # Check exact matches
        if model_name in supported_models:
            return True

        # Check for model families (e.g., any OPT model might be supported)
        if model_name.startswith("facebook/opt-"):
            return True

        # Check for common base models
        if any(base in model_name for base in ["gpt2", "opt-", "DialoGPT"]):
            return True

        return False

    async def get_template_id_by_name(self, template_name: str) -> str:
        """Get template ID by template name using REST API."""
        rest_url = "https://api.runpod.ai/v2/templates"

        response = await self.deps.http_client.get(
            rest_url,
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
        )

        # The response should be a list of templates
        if not isinstance(response, list):
            logger.error(f"Unexpected REST API response format: {type(response)}")
            raise ValueError(f"Invalid REST API response format: {type(response)}")

        templates: list[Dict[str, Any]] = response

        # Find template by name (case-insensitive)
        for template in templates:
            if (
                isinstance(template, dict)
                and template.get("name", "").lower() == template_name.lower()
            ):
                template_id = template.get("id")
                if template_id:
                    return template_id

        # If not found by name, try imageName
        for template in templates:
            if (
                isinstance(template, dict)
                and template.get("imageName", "").lower() == template_name.lower()
            ):
                template_id = template.get("id")
                if template_id:
                    return template_id

        # If still not found, try partial name matching for vLLM templates
        for template in templates:
            if isinstance(template, dict):
                name = template.get("name", "").lower()
                if template_name.lower() in name and ("vllm" in name or "llm" in name):
                    template_id = template.get("id")
                    if template_id:
                        logger.info(
                            f"Found partial match for '{template_name}': '{template.get('name')}' (ID: {template_id})"
                        )
                        return template_id

        available_templates = [f"{t.get('name', 'Unknown')} (ID: {t.get('id', 'Unknown')})" for t in templates[:20]]  # type: ignore  # Show first 20
        if len(templates) > 20:
            available_templates.append(f"... and {len(templates) - 20} more templates")
        raise ValueError(
            f"Template '{template_name}' not found. Available templates: {available_templates}"
        )

    async def create_pod(self, config: Dict[str, Any]) -> str:
        """Create a new RunPod pod with hybrid template/custom image approach.

        Service delivery priority:
        1. use_custom_image=True: Custom Docker image (exact service guarantee)
        2. use_custom_image=False/not set: Try template first (faster), fallback to custom image

        Templates are tried first for supported models/backends to optimize cost and speed,
        with custom images as reliable fallback for full service compatibility.
        """
        rest_url = "https://api.runpod.ai/v2/pod"

        # Explicit custom image request - strict delivery guarantee
        if config.get("use_custom_image", False):
            return await self._create_custom_image_pod(config, rest_url)

        # Hybrid approach: try template first, fallback to custom image
        return await self._create_hybrid_pod(config, rest_url)

    async def _create_hybrid_pod(self, config: Dict[str, Any], rest_url: str) -> str:
        """Try template first, fallback to custom image for maximum compatibility."""
        # If model_name is provided, use hybrid approach
        if "model_name" in config:
            backend = config.get("backend", "vllm")

            # Check if model/backend is supported by templates
            if self.is_model_supported_by_templates(config["model_name"], backend):
                try:
                    logger.info(
                        f"Trying template approach for {config['model_name']} with {backend}"
                    )
                    return await self._create_template_pod(config, rest_url)
                except Exception as e:
                    logger.warning(f"Template approach failed for {config['model_name']}: {e}")
                    logger.info(f"Falling back to custom image for guaranteed service delivery")

            # Fallback to custom image - guaranteed to work
            logger.info(f"Using custom image approach for {config['model_name']} with {backend}")
            return await self._create_custom_image_pod(config, rest_url)

        # If no model_name but template_id provided, use template-only approach
        elif "template_id" in config:
            logger.info(f"Using template-only approach with template_id: {config['template_id']}")
            return await self._create_template_pod(config, rest_url)

        else:
            raise ValueError("Either model_name or template_id is required for pod creation")

    async def _create_custom_image_pod(self, config: Dict[str, Any], rest_url: str) -> str:
        """Create pod with custom Docker image - delivers exact requested service or fails."""
        if "model_name" not in config:
            raise ValueError("model_name is required for custom image pods")

        # Build custom image - this MUST succeed or we fail
        backend = config.get("backend", "vllm")
        try:
            image_name = await self.docker_builder.get_or_build_image(config["model_name"], backend)
        except Exception as e:
            logger.error(
                f"Failed to build custom image for {config['model_name']} with {backend}: {e}"
            )
            raise ValueError(
                f"Cannot create pod: custom image building failed for model '{config['model_name']}' with backend '{backend}'. This service cannot be delivered."
            ) from e

        # Use the successfully built custom image
        pod_data = {
            "imageName": image_name,
            "gpuTypeId": config["gpu_type_id"],
            "gpuCount": config.get("gpu_count", 1),
            "containerDiskInGb": config.get("container_disk_gb", 20),
            "volumeInGb": config.get("volume_gb", 0),
            "startJupyter": config.get("start_jupyter", False),
            "name": config.get(
                "name", f"runpod-llm-{config['model_name'].replace('/', '-')}-{backend}"
            ),
            "env": [],  # Custom images handle their own environment
        }

        # Add optional requirements
        if "min_disk" in config:
            pod_data["minDisk"] = config["min_disk"]
        if "min_memory_gb" in config:
            pod_data["minMemoryInGb"] = config["min_memory_gb"]
        if "min_vcpu_count" in config:
            pod_data["minVcpuCount"] = config["min_vcpu_count"]

        logger.info(f"Creating custom image pod with exact service delivery: {pod_data}")

        try:
            response = await self.deps.http_client.post(
                rest_url,
                json=pod_data,
                headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
            )

            if not response or "id" not in response:
                raise ValueError(f"Invalid API response for custom image pod: {response}")

            pod_id = response["id"]
            logger.info(
                f"Successfully created custom image pod {pod_id} with {config['model_name']} on {backend}"
            )
            return pod_id

        except Exception as e:
            logger.error(f"Failed to create custom image pod: {e}")
            raise

    async def _create_template_pod(self, config: Dict[str, Any], rest_url: str) -> str:
        """Create pod with RunPod template - separate code path from custom images."""
        if "template_id" not in config:
            raise ValueError("template_id is required for template-based pods")

        # Get template ID
        template_input = config["template_id"]
        if template_input.replace("-", "").replace("_", "").isalnum():
            template_id = template_input  # Direct ID
        else:
            template_id = await self.get_template_id_by_name(template_input)

        # Prepare environment variables
        env_vars = []
        if "model_name" in config:
            env_vars.append({"key": "MODEL_NAME", "value": config["model_name"]})
            env_vars.append({"key": "VLLM_MODEL", "value": config["model_name"]})
        if "model_path" in config:
            env_vars.append({"key": "MODEL_PATH", "value": config["model_path"]})

        pod_data = {
            "templateId": template_id,
            "cloudType": "SECURE",
            "gpuTypeId": config["gpu_type_id"],
            "gpuCount": config.get("gpu_count", 1),
            "containerDiskInGb": config.get("container_disk_gb", 20),
            "volumeInGb": config.get("volume_gb", 0),
            "startJupyter": config.get("start_jupyter", False),
            "name": config.get("name", "runpod-llm-template"),
            "env": env_vars,
        }

        # Add optional requirements
        if "min_disk" in config:
            pod_data["minDisk"] = config["min_disk"]
        if "min_memory_gb" in config:
            pod_data["minMemoryInGb"] = config["min_memory_gb"]
        if "min_vcpu_count" in config:
            pod_data["minVcpuCount"] = config["min_vcpu_count"]

        logger.info(f"Creating template-based pod: {pod_data}")

        try:
            response = await self.deps.http_client.post(
                rest_url,
                json=pod_data,
                headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
            )

            if not response or "id" not in response:
                raise ValueError(f"Invalid API response for template pod: {response}")

            pod_id = response["id"]
            logger.info(f"Successfully created template pod {pod_id} with template {template_id}")
            return pod_id

        except Exception as e:
            logger.error(f"Failed to create template pod: {e}")
            raise

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a RunPod pod using REST API."""
        rest_url = f"https://api.runpod.ai/v2/pod/{pod_id}"

        await self.deps.http_client.delete(
            rest_url,
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
        )

    def get_pod_endpoint_url(self, pod_id: str) -> str:
        """Get the endpoint URL for a pod."""
        return f"https://{pod_id}-8000.proxy.runpod.net/v1/chat/completions"

    async def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """Get status of a RunPod pod using REST API."""
        rest_url = f"https://api.runpod.ai/v2/pod/{pod_id}"

        response = await self.deps.http_client.get(
            rest_url,
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
        )

        # Parse ports from the REST API response
        ports = []
        ports_str = response.get("ports")
        if ports_str and isinstance(ports_str, str):
            # REST API returns ports as a string like "8888/http,22/tcp"
            for port_spec in ports_str.split(","):
                if "/" in port_spec:
                    port_num, port_type = port_spec.strip().split("/")
                    try:
                        ports.append({"privatePort": int(port_num), "type": port_type})
                    except ValueError:
                        continue

        return {
            "id": response.get("id", pod_id),
            "status": response.get("desiredStatus", response.get("status", "UNKNOWN")),
            "ip": response.get("ip"),
            "ports": ports,
        }
