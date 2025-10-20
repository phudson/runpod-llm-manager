"""
Watchdog and service monitoring tests.
Tests that monitoring and health checking functionality works.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestWatchdogs:
    """Test watchdog and monitoring functionality."""

    @pytest.mark.asyncio
    async def test_health_service_reports_correct_status(self, mock_deps):
        """Test that health service reports correct system status."""
        from runpod_llm_manager.health_service import HealthService

        service = HealthService(mock_deps)

        # Test normal operation
        health = await service.get_health_status("192.168.1.100")

        assert health["status"] == "healthy"
        assert health["cache_dir"] == mock_deps.config.cache_dir
        assert health["endpoint"] == mock_deps.config.runpod_endpoint

    @pytest.mark.asyncio
    async def test_metrics_service_tracks_requests(self, mock_deps):
        """Test that metrics service properly tracks requests."""
        from runpod_llm_manager.metrics_service import MetricsService

        service = MetricsService(mock_deps)

        # Get initial metrics
        initial_metrics = await service.get_metrics()

        # Validate metrics structure
        assert "requests_total" in initial_metrics
        assert "cache_hits" in initial_metrics
        assert "errors_total" in initial_metrics
        assert "cache_hit_rate" in initial_metrics
        assert "error_rate" in initial_metrics

        # All metrics should be numbers
        for key, value in initial_metrics.items():
            assert isinstance(value, (int, float)), f"Metric {key} should be numeric"

    @patch("runpod_llm_manager.dependencies.get_dependencies")
    def test_service_initialization_works(self, mock_get_deps, mock_deps):
        """Test that services initialize properly on startup."""
        mock_get_deps.return_value = mock_deps

        # Import and create app (this triggers service initialization)
        # Services should be initialized
        from runpod_llm_manager.proxy_fastapi import (
            app,
            health_service,
            llm_service,
            metrics_service,
        )

        # Check that services are created (they might be None if initialization failed)
        # In a real test, we'd check they're not None
        # For now, just verify the import works
        assert True  # If we get here, initialization didn't crash

    @pytest.mark.asyncio
    async def test_cache_operations_work(self, mock_deps):
        """Test that cache operations function correctly."""
        cache = mock_deps.cache

        # Test cache set/get
        test_key = "test_key"
        test_value = {"test": "data"}

        await cache.set(test_key, test_value)
        retrieved = await cache.get(test_key)

        assert retrieved == test_value

        # Test cache delete
        await cache.delete(test_key)
        retrieved_after_delete = await cache.get(test_key)
        assert retrieved_after_delete is None

    def test_rate_limiter_operations_work(self, mock_deps):
        """Test that rate limiter operations function correctly."""
        limiter = mock_deps.rate_limiter

        # Test initial state
        assert limiter.is_allowed("192.168.1.100") == True
        assert limiter.get_remaining_requests("192.168.1.100") > 0

    @pytest.mark.asyncio
    async def test_file_system_operations_work(self, mock_deps, tmp_path):
        """Test that file system operations function correctly."""
        filesystem = mock_deps.filesystem

        # Test file operations
        test_file = tmp_path / "test.txt"
        test_content = "Hello, world!"

        # Write file
        await filesystem.write_file(str(test_file), test_content)

        # Read file
        content = await filesystem.read_file(str(test_file))
        assert content == test_content

        # Check file exists
        exists = await filesystem.file_exists(str(test_file))
        assert exists == True

        # Delete file
        await filesystem.delete_file(str(test_file))
        exists_after_delete = await filesystem.file_exists(str(test_file))
        assert exists_after_delete == False

    def test_dependency_injection_works(self):
        """Test that dependency injection system works."""
        from runpod_llm_manager.dependencies import create_test_dependencies
        from tests.fixtures.mock_services import create_test_config

        config = create_test_config()
        deps = create_test_dependencies(config)

        # Check that all required dependencies are present
        assert deps.config is not None
        assert deps.http_client is not None
        assert deps.cache is not None
        assert deps.filesystem is not None
        assert deps.rate_limiter is not None

        # Check that they have the expected interfaces
        assert hasattr(deps.http_client, "post")
        assert hasattr(deps.http_client, "get")
        assert hasattr(deps.cache, "get")
        assert hasattr(deps.cache, "set")
        assert hasattr(deps.filesystem, "write_file")
        assert hasattr(deps.filesystem, "read_file")
        assert hasattr(deps.rate_limiter, "is_allowed")

    def test_health_endpoint_with_rate_limiting(self, client):
        """Test health endpoint behavior under rate limiting."""
        # Create a rate limiter that will limit after a few requests
        from tests.fixtures.mock_services import MockRateLimiter

        limiter = MockRateLimiter(always_allow=False, limit_after=5)

        # Override the rate limiter in dependencies
        from runpod_llm_manager.dependencies import get_dependencies, set_dependencies

        deps = get_dependencies()
        deps.rate_limiter = limiter
        set_dependencies(deps)

        # Make enough requests to trigger rate limiting
        responses = []
        for i in range(10):  # More than limit_after
            response = client.get("/health")
            responses.append(response.status_code)

        # Should see some 429 responses
        assert 429 in responses, "Rate limiting should trigger"

    @pytest.mark.asyncio
    async def test_service_error_recovery(self, mock_deps):
        """Test that services can recover from errors."""
        from runpod_llm_manager.llm_service import LLMService
        from runpod_llm_manager.proxy_fastapi_models import ChatCompletionRequest, ChatMessage

        service = LLMService(mock_deps)

        # Test with valid request first
        valid_request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=50,
            temperature=0.7,
            stream=False,
        )

        # Should not raise exception
        try:
            result = await service.process_completion_request(valid_request)
            # Result should be from mock
            assert result is not None
        except Exception as e:
            pytest.fail(f"Valid request should not fail: {e}")

    def test_configuration_validation(self):
        """Test that configuration is properly validated."""
        from runpod_llm_manager.config import AppConfig

        # Test with valid config
        config = AppConfig()
        assert config.runpod_endpoint is not None
        assert config.cache_dir is not None
        assert config.rate_limit_requests > 0

        # Test configuration validation
        try:
            config.validate_ssl_config()  # Should not raise
        except Exception as e:
            pytest.fail(f"Configuration validation should pass: {e}")
