"""
Core functional tests for RunPod LLM Manager.
Tests that validate the basic functionality works correctly.
"""

from unittest.mock import patch

import pytest


class TestCoreFunctionality:
    """Test core functional requirements."""

    def test_fastapi_proxy_imports_successfully(self):
        """Test that FastAPI proxy can be imported without errors."""
        # This tests that all imports work and basic app structure exists
        from runpod_llm_manager.proxy_fastapi import app

        # Test that app is a FastAPI instance
        assert hasattr(app, "routes")
        assert len(app.routes) > 0

        # Test that we have some routes (basic smoke test)
        assert len(app.routes) >= 5  # health, metrics, dashboard, chat completions, streaming

    def setup_method(self):
        """Set up test dependencies before each test."""
        from runpod_llm_manager.dependencies import set_dependencies
        from tests.fixtures.mock_services import (
            create_chat_completion_response,
            create_mock_dependencies,
            create_test_config,
        )

        config = create_test_config()
        # Create mock dependencies with chat completion response
        http_responses = {
            "http://mock-runpod:4010/v1/chat/completions": create_chat_completion_response()
        }
        mock_deps = create_mock_dependencies(config, http_responses)
        set_dependencies(mock_deps)

    def test_health_endpoint_returns_expected_structure(self, client):
        """Test /health endpoint returns proper structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Validate health response structure
        assert "status" in data
        assert "timestamp" in data
        assert "cache_dir" in data
        assert "endpoint" in data
        assert "security" in data

        # Validate security info
        security = data["security"]
        assert "rate_limit_remaining" in security
        assert "rate_limit_limit" in security

    def test_chat_completions_endpoint_accepts_valid_requests(self, client):
        """Test /v1/chat/completions accepts and processes valid requests."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, test message"}],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate OpenAI-compatible response
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0

        choice = data["choices"][0]
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]

    def test_rate_limiting_works(self, client):
        """Test that rate limiting is functional."""
        from runpod_llm_manager.dependencies import get_dependencies, set_dependencies
        from tests.fixtures.mock_services import MockRateLimiter

        # Create a rate limiter that limits after 3 requests
        limiting_limiter = MockRateLimiter(always_allow=False, limit_after=3)
        deps = get_dependencies()
        deps.rate_limiter = limiting_limiter
        set_dependencies(deps)

        # Make multiple requests
        responses = []
        for i in range(6):  # More than the limit
            response = client.get("/health")
            responses.append(response.status_code)

        # Should have some successful (200) and some rate limited (429)
        assert 200 in responses  # At least some succeeded
        assert 429 in responses  # At least some were rate limited

    def test_security_headers_applied(self, client):
        """Test that security headers are applied."""
        response = client.get("/health")

        # Check security headers are present
        headers = response.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("X-XSS-Protection") == "1; mode=block"

    @patch("runpod_llm_manager.llm_service.LLMService.process_completion_request")
    def test_proxy_handles_service_errors_gracefully(self, mock_process, client):
        """Test that proxy handles LLM service errors gracefully."""
        # Mock service to raise an exception
        mock_process.side_effect = Exception("LLM service error")

        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}

        response = client.post("/v1/chat/completions", json=payload)

        # Should return 500 error, not crash
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_input_validation_works(self, client):
        """Test that input validation rejects invalid requests."""
        # Test empty messages
        response = client.post(
            "/v1/chat/completions", json={"model": "gpt-3.5-turbo", "messages": []}
        )
        assert response.status_code == 422  # Validation error

        # Test invalid model
        response = client.post(
            "/v1/chat/completions",
            json={"model": "", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 422

    def test_cors_headers_present(self, client):
        """Test that CORS headers are applied."""
        # Make a regular request with Origin header
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "access-control-allow-credentials" in response.headers

    def test_dashboard_endpoint_provides_system_info(self, client):
        """Test /dashboard provides comprehensive system information."""
        response = client.get("/dashboard")

        assert response.status_code == 200
        data = response.json()

        assert "health" in data
        assert "metrics" in data
        assert "system" in data
        assert "configuration" in data

        # Validate system info
        system = data["system"]
        assert "python_version" in system
        assert "platform" in system

    def test_metrics_endpoint_returns_data(self, client):
        """Test /metrics endpoint returns performance data."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        # Validate metrics structure
        assert "requests_total" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "error_rate" in data

    def test_streaming_endpoint_exists(self, client):
        """Test that streaming endpoint exists (even if it delegates)."""
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}

        response = client.post("/v1/chat/completions/stream", json=payload)

        # Should work (currently delegates to regular endpoint)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_streaming_response_format(self, client):
        """Test that streaming endpoint returns proper SSE format."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        response = client.post("/v1/chat/completions/stream", json=payload)

        # Should return streaming response
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/plain"
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Check that response contains SSE data
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content

    def test_streaming_response_chunks(self, client):
        """Test that streaming response contains proper chunk structure."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        response = client.post("/v1/chat/completions/stream", json=payload)

        assert response.status_code == 200
        content = response.text

        # Should contain multiple data chunks
        lines = content.strip().split("\n\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        assert len(data_lines) > 1  # Should have at least initial and final chunks

        # Check first chunk structure
        import json

        first_chunk = json.loads(data_lines[0].replace("data: ", ""))
        assert "id" in first_chunk
        assert "object" in first_chunk
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "choices" in first_chunk

        # Check final chunk
        assert data_lines[-1] == "data: [DONE]"

    def test_streaming_error_handling(self, client):
        """Test that streaming endpoint handles errors properly."""
        # Test with invalid payload that should cause an error
        payload = {"model": "", "messages": [], "stream": True}

        response = client.post("/v1/chat/completions/stream", json=payload)

        # Should still return 200 but with error in stream
        assert response.status_code == 200
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content
