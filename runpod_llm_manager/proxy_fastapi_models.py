"""
Pydantic models for RunPod LLM Manager FastAPI proxy
Input validation and data models
"""

from pydantic import BaseModel, Field, field_validator

try:
    from typing import List, Annotated
except ImportError:
    # Python 3.8 compatibility
    from typing import List
    from typing_extensions import Annotated
import re


class ChatMessage(BaseModel):
    """Model for individual chat messages."""

    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v):
        """Sanitize message content to prevent injection attacks."""
        # Remove potentially harmful patterns
        v = re.sub(r"<script[^>]*>.*?</script>", "", v, flags=re.IGNORECASE)
        v = re.sub(r"javascript:", "", v, flags=re.IGNORECASE)
        return v


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""

    model: str = Field(..., min_length=1, max_length=100)
    messages: Annotated[List[ChatMessage], Field(min_length=1, max_length=50)]
    max_tokens: int = Field(10, ge=1, le=4096)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    stream: bool = Field(False)


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str
    timestamp: float
    cache_dir: str
    endpoint: str
    cache_entries: int
    cache_size_bytes: int
    max_cache_size: int
    max_cache_bytes: int
    uptime: float
    security: dict


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    requests_total: int
    cache_hits: int
    cache_misses: int
    errors_total: int
    avg_response_time: float
    cache_entries: int
    cache_size_bytes: int
    cache_hit_rate: float
    error_rate: float


class DashboardResponse(BaseModel):
    """Response model for dashboard."""

    health: HealthResponse
    metrics: MetricsResponse
    system: dict
    configuration: dict
