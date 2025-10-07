"""
Configuration management for RunPod LLM Manager
Uses Pydantic BaseSettings for type-safe configuration with environment variable support
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class AppConfig(BaseSettings):
    """Application configuration with environment variable support."""

    # API Settings
    runpod_endpoint: str = Field(
        default="http://localhost:8000/v1/chat/completions",
        description="RunPod API endpoint URL",
    )
    runpod_api_key: Optional[str] = Field(
        default=None, description="RunPod API key (required for production)"
    )

    # Cache Settings
    cache_dir: str = Field(default="/tmp/llm_cache", description="Directory for cache files")
    max_cache_size: int = Field(
        default=1000, description="Maximum number of cached responses", ge=1
    )
    cache_size_bytes: int = Field(
        default=1073741824, description="Maximum cache size in bytes", ge=1024  # 1GB
    )

    # Security Settings
    rate_limit_requests: int = Field(
        default=60, description="Number of requests allowed per time window", ge=1
    )
    rate_limit_window: int = Field(
        default=60, description="Rate limit time window in seconds", ge=1
    )
    max_request_size: int = Field(
        default=1048576, description="Maximum request size in bytes", ge=1024  # 1MB
    )
    max_content_length: int = Field(
        default=50000, description="Maximum content length for messages", ge=100  # 50KB
    )

    # Feature Flags
    enable_profiling: bool = Field(default=False, description="Enable debug profiling endpoints")
    use_https: bool = Field(default=False, description="Enable HTTPS mode")

    # SSL Settings
    ssl_cert: Optional[str] = Field(default=None, description="Path to SSL certificate file")
    ssl_key: Optional[str] = Field(default=None, description="Path to SSL private key file")

    # CORS Settings
    allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080",
        description="Comma-separated list of allowed CORS origins",
    )

    # Test/Development Settings
    test_mode: bool = Field(default=False, description="Enable test mode (disables real API calls)")
    mock_runpod_url: Optional[str] = Field(default=None, description="Mock RunPod URL for testing")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    def validate_ssl_config(self) -> None:
        """Validate SSL configuration consistency."""
        if self.use_https:
            if not self.ssl_cert or not self.ssl_key:
                raise ValueError("HTTPS enabled but ssl_cert or ssl_key not provided")
            if not os.path.exists(self.ssl_cert) or not os.path.exists(self.ssl_key):
                raise ValueError("SSL certificate or key file not found")

    def validate_api_key(self) -> None:
        """Validate API key is present when not in test mode."""
        if not self.test_mode and not self.runpod_api_key:
            raise ValueError("RUNPOD_API_KEY environment variable not set")


# Enable test mode for pytest before creating config
test_mode_override = os.getenv("PYTEST_CURRENT_TEST") is not None

# Global config instance
config = AppConfig()

# Override test mode if running pytest
if test_mode_override:
    config.test_mode = True

# Validate configuration on import (skip API key validation for testing)
try:
    config.validate_ssl_config()
    # Skip API key validation for testing - will be validated at runtime when needed
    # if not test_mode_override and not config.test_mode:
    #     config.validate_api_key()
except ValueError as e:
    # Only raise error if not in test environment
    if not test_mode_override and not config.test_mode:
        print(f"❌ Configuration error: {e}")
        raise
    else:
        print(f"⚠️ Configuration warning (ignored in test mode): {e}")
