"""
Functional test configuration and fixtures.
Functional tests use real dependencies and may incur costs.
"""

import os

import pytest

from runpod_llm_manager.config import AppConfig
from runpod_llm_manager.dependencies import (
    AIOFilesFileSystem,
    Dependencies,
    HTTPXClient,
    InMemoryCache,
    InMemoryRateLimiter,
)
from tests.fixtures.mock_services import create_mock_dependencies


@pytest.fixture(params=["real"])  # Can be extended to ["mock", "real"] if needed
def deps_type(request):
    """Parameter to choose between mock and real dependencies for functional tests."""
    return request.param


@pytest.fixture
def test_config(deps_type):
    """Configuration for testing - mock or real."""
    if deps_type == "mock":
        from tests.fixtures.mock_services import create_test_config

        return create_test_config()
    elif deps_type == "real":
        api_key = os.getenv("RUNPOD_API_KEY") or os.getenv("RUNPROD_API_KEY")
        if not api_key:
            pytest.skip(
                "RUNPOD_API_KEY or RUNPROD_API_KEY environment variable required for functional tests"
            )

        return AppConfig(
            runpod_endpoint=os.getenv("RUNPOD_ENDPOINT", "https://api.runpod.ai"),
            runpod_api_key=api_key,
            cache_dir=os.getenv("CACHE_DIR", "./cache"),
            max_cache_size=int(os.getenv("MAX_CACHE_SIZE", "1000")),
            cache_size_bytes=int(
                os.getenv("CACHE_SIZE_BYTES", str(10 * 1024 * 1024 * 1024))
            ),  # 10GB
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            use_https=os.getenv("USE_HTTPS", "true").lower() == "true",
        )


@pytest.fixture
def test_deps(test_config, deps_type):
    """Dependencies for testing - mock or real."""
    if deps_type == "mock":
        return create_mock_dependencies(config=test_config)
    elif deps_type == "real":
        return Dependencies(
            config=test_config,
            http_client=HTTPXClient(timeout=60.0),  # Longer timeout for real operations
            cache=InMemoryCache(),
            rate_limiter=InMemoryRateLimiter(
                requests_per_window=test_config.rate_limit_requests,
                window_seconds=test_config.rate_limit_window,
            ),
            filesystem=AIOFilesFileSystem(),
        )


@pytest.fixture
def test_services(test_deps):
    """Service instances for testing."""
    from runpod_llm_manager.docker_builder import DockerImageBuilderService
    from runpod_llm_manager.llm_service import LLMService
    from runpod_llm_manager.pod_service import PodManagementService

    return {
        "docker_builder": DockerImageBuilderService(test_deps),
        "pod_service": PodManagementService(test_deps),
        "llm_service": LLMService(test_deps),
    }
