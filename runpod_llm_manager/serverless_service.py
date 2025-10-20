"""
Serverless Service
Handles RunPod serverless endpoint management with vLLM and ModelStore optimization.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .dependencies import Dependencies

logger = logging.getLogger(__name__)


class ServerlessService:
    """Service for RunPod serverless endpoint management with vLLM and ModelStore optimization."""

    def __init__(self, deps: Dependencies):
        self.deps = deps

    async def create_vllm_endpoint(self, config: Dict[str, Any]) -> str:
        """Create a vLLM serverless endpoint using REST API."""
        serverless_url = "https://api.runpod.ai/v2/endpoint"

        endpoint_data = {
            "name": config.get("name", "vllm-llm-endpoint"),
            "templateId": config["template_id"],
            "gpuTypeId": config["gpu_type_id"],
            "gpuCount": config.get("gpu_count", 1),
            "workerCount": config.get("worker_count", 1),
            "minWorkers": config.get("min_workers", 0),
            "maxWorkers": config.get("max_workers", 3),
            "idleTimeout": config.get("idle_timeout", 300),
            "locations": config.get("locations", ""),
            "networkVolumeId": config.get("network_volume_id"),
            "dataCenterId": config.get("data_center_id"),
            "imageName": config.get("image_name", "runpod/vllm:latest"),
            "containerDiskInGb": config.get("container_disk_gb", 20),
            "env": self._prepare_vllm_env_vars(config),
            "dockerArgs": config.get("docker_args", ""),
            "ports": config.get("ports", "8000/http"),
        }

        response = await self.deps.http_client.post(
            serverless_url,
            json=endpoint_data,
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
        )

        if not response or "id" not in response:
            logger.error(f"Invalid REST API response: {response}")
            raise ValueError(f"Invalid REST API response: {response}")

        return response["id"]

    def _prepare_vllm_env_vars(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare environment variables for vLLM with ModelStore optimization."""
        env_vars = []

        # Model configuration
        if "model_name" in config:
            env_vars.append({"key": "MODEL_NAME", "value": config["model_name"]})
        if "model_path" in config:
            env_vars.append({"key": "MODEL_PATH", "value": config["model_path"]})

        # vLLM specific settings
        env_vars.extend(
            [
                {"key": "VLLM_MODEL", "value": config.get("model_name", "facebook/opt-125m")},
                {
                    "key": "VLLM_TENSOR_PARALLEL_SIZE",
                    "value": str(config.get("tensor_parallel_size", 1)),
                },
                {
                    "key": "VLLM_GPU_MEMORY_UTILIZATION",
                    "value": str(config.get("gpu_memory_utilization", 0.9)),
                },
                {"key": "VLLM_MAX_MODEL_LEN", "value": str(config.get("max_model_len", 4096))},
                {
                    "key": "VLLM_ENFORCE_EAGER",
                    "value": str(config.get("enforce_eager", False)).lower(),
                },
            ]
        )

        # ModelStore optimization
        if config.get("use_model_store", True):
            env_vars.extend(
                [
                    {"key": "RUNPOD_USE_MODEL_STORE", "value": "true"},
                    {"key": "MODEL_STORE_CACHE_DIR", "value": "/runpod-volume/model-cache"},
                    {
                        "key": "MODEL_STORE_ENDPOINT",
                        "value": config.get("model_store_endpoint", "https://api.runpod.io/v2"),
                    },
                ]
            )

        # Add any additional environment variables
        if "env" in config:
            env_vars.extend(config["env"])

        return env_vars

    def get_endpoint_url(self, endpoint_id: str) -> str:
        """Get the endpoint URL for a serverless endpoint."""
        return f"https://{endpoint_id}.runpod.net/v1/chat/completions"

    async def get_endpoint_status(self, endpoint_id: str) -> Dict[str, Any]:
        """Get status of a serverless endpoint using REST API."""
        rest_url = f"https://api.runpod.ai/v2/endpoint/{endpoint_id}"

        response = await self.deps.http_client.get(
            rest_url,
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
        )

        return response

    async def delete_endpoint(self, endpoint_id: str) -> None:
        """Delete a serverless endpoint using REST API."""
        rest_url = f"https://api.runpod.ai/v2/endpoint/{endpoint_id}"

        await self.deps.http_client.delete(
            rest_url,
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
        )

    async def get_model_store_status(self, model_name: str) -> Dict[str, Any]:
        """Check if a model is cached in ModelStore by querying the API."""
        try:
            # Query ModelStore API to check model status
            modelstore_url = f"{self.deps.config.runpod_endpoint}/model-store/status/{model_name}"

            response = await self.deps.http_client.get(
                modelstore_url,
                headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"},
            )

            # Return the actual API response
            return response

        except Exception as e:
            logger.warning(f"Failed to query ModelStore API for {model_name}: {e}")
            # Return a fallback status indicating model is not cached
            return {
                "model_name": model_name,
                "cached": False,
                "error": str(e),
                "cache_location": None,
                "last_accessed": None,
                "size_gb": None,
            }
