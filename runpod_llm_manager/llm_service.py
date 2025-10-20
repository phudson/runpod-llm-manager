"""
LLM Service
Handles LLM completion operations with caching and endpoint management.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from .config import AppConfig
from .dependencies import Dependencies
from .proxy_fastapi_models import ChatCompletionRequest, ChatMessage

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM completion operations with support for both pod-based and serverless endpoints."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.pod_service = None  # Will be set by caller if needed
        self.serverless_service = None  # Will be set by caller if needed
        self.docker_builder = None  # Will be set by caller if needed
        self.endpoint_url: Optional[str] = None
        self._metrics_service = None  # For recording cache metrics

    def set_services(self, pod_service=None, serverless_service=None, docker_builder=None):
        """Set related services for endpoint URL resolution."""
        self.pod_service = pod_service
        self.serverless_service = serverless_service
        self.docker_builder = docker_builder

    def set_metrics_service(self, metrics_service):
        """Set metrics service for recording cache statistics."""
        self._metrics_service = metrics_service

    def set_endpoint_url(self, url: str) -> None:
        """Set the endpoint URL for LLM requests."""
        self.endpoint_url = url

    async def process_completion_request(
        self,
        request: ChatCompletionRequest,
        endpoint_url: Optional[str] = None,
        mode: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        pod_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a chat completion request with caching.

        Args:
            request: The chat completion request
            endpoint_url: The pre-calculated endpoint URL to use
            mode: Optional mode ("serverless" or "pod") to determine endpoint URL
            endpoint_id: Serverless endpoint ID when mode="serverless"
            pod_id: Pod ID when mode="pod"
        """
        # For streaming requests, skip caching and go directly to endpoint
        if getattr(request, "stream", False):
            if endpoint_url is None:
                raise ValueError(
                    "endpoint_url is required for streaming requests - config endpoint is not valid for dynamic endpoints"
                )
            return await self._call_endpoint(request, endpoint_url)

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Try cache first
        cached_result = await self.deps.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key: {cache_key[:8]}...")
            # Record cache hit in metrics
            if hasattr(self, "_metrics_service") and self._metrics_service:
                self._metrics_service.record_cache_hit()
            return cached_result

        # Cache miss - call the endpoint
        logger.info(f"Cache miss for key: {cache_key[:8]}...")
        # Record cache miss in metrics
        if hasattr(self, "_metrics_service") and self._metrics_service:
            self._metrics_service.record_cache_miss()

        if endpoint_url is None:
            # Determine endpoint URL based on mode
            if mode == "serverless" and endpoint_id and self.serverless_service:
                endpoint_url = self.serverless_service.get_endpoint_url(endpoint_id)
            elif mode == "pod" and pod_id and self.pod_service:
                endpoint_url = self.pod_service.get_pod_endpoint_url(pod_id)
            else:
                raise ValueError(
                    "endpoint_url is required for LLM requests - config endpoint is not valid for dynamic endpoints"
                )

        if endpoint_url is None:
            raise ValueError(
                "endpoint_url is required for LLM requests - config endpoint is not valid for dynamic endpoints"
            )

        response = await self._call_endpoint(request, endpoint_url)

        # Cache the result
        await self.deps.cache.set(cache_key, response)

        logger.info(f"Cached new response for key: {cache_key[:8]}...")
        return response

    async def _call_endpoint(self, request, endpoint_url: str) -> Dict[str, Any]:
        """Call an endpoint with the given URL."""
        # Handle both ChatCompletionRequest objects and dicts
        if hasattr(request, "model_dump"):
            payload = request.model_dump()
        else:
            payload = request
        return await self.deps.http_client.post(endpoint_url, json=payload)

    def _generate_cache_key(self, request) -> str:
        """Generate deterministic cache key from request."""
        # Handle both ChatCompletionRequest objects and dicts
        if hasattr(request, "model_dump"):
            request_dict = request.model_dump()
        else:
            request_dict = request

        # Sort keys for consistent hashing
        normalized = json.dumps(request_dict, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()
