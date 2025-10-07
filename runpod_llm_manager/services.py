"""
Business logic services for RunPod LLM Manager
Extracted from HTTP handlers for better testability and separation of concerns
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone

from .config import AppConfig
from .dependencies import Dependencies
from .proxy_fastapi_models import ChatCompletionRequest, ChatMessage

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM completion operations."""

    def __init__(self, deps: Dependencies):
        self.deps = deps

    async def process_completion_request(
        self,
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """Process a chat completion request with caching."""
        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Try cache first
        cached_result = await self.deps.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key: {cache_key[:8]}...")
            return cached_result

        # Cache miss - call external API
        logger.info(f"Cache miss for key: {cache_key[:8]}...")

        # Prepare request payload
        payload = request.model_dump()

        # Make API call
        response = await self.deps.http_client.post(
            self.deps.config.runpod_endpoint,
            json=payload
        )

        # Cache the result
        await self.deps.cache.set(cache_key, response)

        logger.info(f"Cached new response for key: {cache_key[:8]}...")
        return response

    def _generate_cache_key(self, request: ChatCompletionRequest) -> str:
        """Generate deterministic cache key from request."""
        # Create a normalized representation for consistent hashing
        request_dict = request.model_dump()
        # Sort keys for consistent hashing
        normalized = json.dumps(request_dict, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()


class PodManagementService:
    """Service for RunPod pod management operations."""

    def __init__(self, deps: Dependencies):
        self.deps = deps

    async def create_pod(self, config: Dict[str, Any]) -> str:
        """Create a new RunPod pod."""
        mutation = """
        mutation StartPod($input: PodInput!) {
          podCreate(input: $input) {
            id
          }
        }
        """

        input_data = {
            "templateId": config["template_id"],
            "cloudType": "SECURE",
            "gpuTypeId": config["gpu_type_id"],
            "modelId": config["modelStoreId"],
            "containerDiskInGb": 20,
            "volumeInGb": 0,
            "startJupyter": False,
            "name": "runpod-llm"
        }

        response = await self.deps.http_client.post(
            self.deps.config.runpod_endpoint.replace("/v1/chat/completions", "/graphql"),
            json={"query": mutation, "variables": {"input": input_data}},
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"}
        )

        return response["data"]["podCreate"]["id"]

    async def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """Get status of a RunPod pod."""
        query = """
        query PodStatus($id: ID!) {
          pod(id: $id) {
            id
            status
            ip
            ports {
              ip
              privatePort
              publicPort
            }
          }
        }
        """

        response = await self.deps.http_client.post(
            self.deps.config.runpod_endpoint.replace("/v1/chat/completions", "/graphql"),
            json={"query": query, "variables": {"id": pod_id}},
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"}
        )

        return response["data"]["pod"]

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a RunPod pod."""
        mutation = """
        mutation TerminatePod($id: ID!) {
          podTerminate(id: $id) {
            id
          }
        }
        """

        await self.deps.http_client.post(
            self.deps.config.runpod_endpoint.replace("/v1/chat/completions", "/graphql"),
            json={"query": mutation, "variables": {"id": pod_id}},
            headers={"Authorization": f"Bearer {self.deps.config.runpod_api_key}"}
        )


class HealthService:
    """Service for health check operations."""

    def __init__(self, deps: Dependencies):
        self.deps = deps

    async def get_health_status(self, client_ip: str = "unknown") -> Dict[str, Any]:
        """Get comprehensive health status."""
        # Get cache statistics
        # Note: In a real implementation, we'd need to extend the cache protocol
        # to provide statistics. For now, returning basic info.

        remaining_requests = self.deps.rate_limiter.get_remaining_requests(client_ip)

        return {
            "status": "healthy",
            "timestamp": datetime.now().timestamp(),
            "cache_dir": self.deps.config.cache_dir,
            "endpoint": self.deps.config.runpod_endpoint,
            "cache_entries": 0,  # Placeholder
            "cache_size_bytes": 0,  # Placeholder
            "max_cache_size": self.deps.config.max_cache_size,
            "max_cache_bytes": self.deps.config.cache_size_bytes,
            "uptime": 0,  # Placeholder - would need app start time
            "security": {
                "rate_limit_remaining": remaining_requests,
                "rate_limit_limit": self.deps.config.rate_limit_requests,
                "rate_limit_window_seconds": self.deps.config.rate_limit_window,
                "https_enabled": self.deps.config.use_https,
                "cors_enabled": True
            }
        }


class MetricsService:
    """Service for metrics collection and reporting."""

    def __init__(self, deps: Dependencies):
        self.deps = deps

    async def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics."""
        # Placeholder implementation
        # In a real implementation, this would collect actual metrics
        return {
            "requests_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors_total": 0,
            "avg_response_time": 0.0,
            "cache_entries": 0,
            "cache_size_bytes": 0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }


# Service factory functions

def create_llm_service(deps: Optional[Dependencies] = None) -> LLMService:
    """Create LLM service instance."""
    if deps is None:
        from .dependencies import get_default_dependencies
        deps = get_default_dependencies()
    return LLMService(deps)


def create_pod_management_service(deps: Optional[Dependencies] = None) -> PodManagementService:
    """Create pod management service instance."""
    if deps is None:
        from .dependencies import get_default_dependencies
        deps = get_default_dependencies()
    return PodManagementService(deps)


def create_health_service(deps: Optional[Dependencies] = None) -> HealthService:
    """Create health service instance."""
    if deps is None:
        from .dependencies import get_default_dependencies
        deps = get_default_dependencies()
    return HealthService(deps)


def create_metrics_service(deps: Optional[Dependencies] = None) -> MetricsService:
    """Create metrics service instance."""
    if deps is None:
        from .dependencies import get_default_dependencies
        deps = get_default_dependencies()
    return MetricsService(deps)