"""
Docker Image Builder Service
Handles building and pushing custom Docker images for LLM backends.
"""

import hashlib
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict

from .dependencies import Dependencies

logger = logging.getLogger(__name__)


class DockerImageBuilderService:
    """Service for building custom Docker images on-demand using registry tags."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        # Use Docker Hub registry from environment or default to runpod
        self.registry = os.getenv("DOCKER_USERNAME", "runpod")
        self.repository = "llm-images"

    async def get_or_build_image(self, model_name: str, backend: str = "vllm") -> str:
        """Get existing image or build new one for the specified model and backend."""
        # Generate deterministic image tag based on model and backend
        image_tag = self._generate_image_tag(model_name, backend)
        full_image_name = f"{self.registry}/{self.repository}:{image_tag}"

        # Check if image already exists in registry
        if await self._image_exists_in_registry(full_image_name):
            logger.info(f"Using existing image: {full_image_name}")
            return full_image_name

        # Build and push new image
        logger.info(f"Building new image: {full_image_name}")
        await self._build_and_push_image(model_name, backend, full_image_name)

        return full_image_name

    def _generate_image_tag(self, model_name: str, backend: str) -> str:
        """Generate deterministic image tag."""
        # Create hash of model + backend for uniqueness
        content = f"{model_name}:{backend}".encode()
        hash_suffix = hashlib.sha256(content).hexdigest()[:12]
        # Clean model name for Docker tag
        clean_model = model_name.replace("/", "-").replace(":", "-").lower()
        return f"{backend}-{clean_model}-{hash_suffix}"

    async def _image_exists_in_registry(self, image_name: str) -> bool:
        """Check if image exists in Docker registry using Docker Hub API with caching."""
        # Check cache first
        cache_key = f"registry_check_{image_name}"
        cached_result = await self.deps.cache.get(cache_key)
        if cached_result is not None:
            return cached_result.get("exists", False)

        try:
            # Extract registry, repository, and tag from image name
            # Format: registry/repository:tag
            if ":" not in image_name:
                result = False
            else:
                registry_repo, tag = image_name.split(":", 1)
                if "/" not in registry_repo:
                    result = False
                else:
                    registry, repository = registry_repo.split("/", 1)

                    # For Docker Hub (registry = "runpod"), use the API
                    if registry == "runpod" or registry == "index.docker.io":
                        # Docker Hub API to check if tag exists
                        api_url = f"https://index.docker.io/v2/{repository}/manifests/{tag}"
                        response = await self.deps.http_client.get(api_url)
                        result = response is not None and "errors" not in response
                    else:
                        # For other registries, we'd need different API calls
                        # For now, assume it exists if we can't check
                        logger.warning(
                            f"Cannot verify image existence for registry {registry}, assuming it exists"
                        )
                        result = True

        except Exception as e:
            logger.warning(f"Error checking image existence: {e}")
            # If we can't check, assume it doesn't exist so we build it
            result = False

        # Cache the result for future checks
        await self.deps.cache.set(cache_key, {"exists": result})
        return result

    async def _build_and_push_image(self, model_name: str, backend: str, image_name: str) -> None:
        """Build and push the Docker image to registry."""
        if backend.lower() == "vllm":
            dockerfile_content = self._generate_vllm_dockerfile(model_name)
        elif backend.lower() == "ollama":
            dockerfile_content = self._generate_ollama_dockerfile(model_name)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        await self._build_and_push_docker_image(dockerfile_content, image_name, backend, model_name)

    async def _build_and_push_docker_image(
        self, dockerfile_content: str, image_name: str, backend: str, model_name: str
    ) -> None:
        """Build and push Docker image with common logic."""
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(
                f"Building and pushing {backend} image: {image_name} for model: {model_name}"
            )

            # Write Dockerfile to temporary directory
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            logger.info(
                f"Dockerfile written to {dockerfile_path} ({len(dockerfile_content)} chars)"
            )

            try:
                # Build Docker image
                logger.info(f"Running: docker build -t {image_name} {temp_dir}")
                build_result = subprocess.run(
                    ["docker", "build", "-t", image_name, temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes timeout
                )

                if build_result.returncode != 0:
                    logger.error(f"Docker build failed: {build_result.stderr}")
                    raise RuntimeError(f"Docker build failed: {build_result.stderr}")

                logger.info("Docker build completed successfully")

                # Push Docker image
                logger.info(f"Running: docker push {image_name}")
                push_result = subprocess.run(
                    ["docker", "push", image_name],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout
                )

                if push_result.returncode != 0:
                    logger.error(f"Docker push failed: {push_result.stderr}")
                    raise RuntimeError(f"Docker push failed: {push_result.stderr}")

                logger.info(f"Successfully built and pushed {backend} image: {image_name}")

            except subprocess.TimeoutExpired:
                logger.error(f"Docker operation timed out for image: {image_name}")
                raise RuntimeError(f"Docker operation timed out for image: {image_name}")
            except FileNotFoundError:
                logger.error(
                    "Docker command not found. Please ensure Docker is installed and accessible."
                )
                raise RuntimeError(
                    "Docker command not found. Please ensure Docker is installed and accessible."
                )
            except Exception as e:
                logger.error(f"Failed to build/push Docker image {image_name}: {e}")
                raise

    def _generate_vllm_dockerfile(self, model_name: str) -> str:
        """Generate Dockerfile for vLLM-based image."""
        return f"""# Use RunPod's PyTorch base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables for vLLM
ENV PYTHONUNBUFFERED="1" \\
    RUNPOD_LLM_BACKEND="vllm" \\
    RUNPOD_LLM_MODEL_NAME="{model_name}" \\
    VLLM_MODEL="{model_name}"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    git \\
    python3 \\
    python3-pip && \\
    rm -rf /var/lib/apt/lists/*

# Install vLLM and other dependencies
RUN pip3 install --root-user-action=ignore --no-cache-dir \\
    vllm \\
    runpod \\
    requests \\
    transformers \\
    accelerate \\
    torch \\
    torchvision \\
    torchaudio

# Skip model pre-download for now - will download at runtime
# RUN python3 -c "
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = os.environ.get('VLLM_MODEL', 'facebook/opt-125m')
# print(f'Pre-downloading model: {model_name}')
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#     print(f'Successfully downloaded model: {model_name}')
# except Exception as e:
#     print(f'Warning: Could not pre-download model {model_name}:', e)
#     print('Model will be downloaded at runtime')
# "

# Copy custom vLLM server script
COPY <<EOF /app/start_vllm.py
#!/usr/bin/env python3
import os
import sys
import subprocess

if __name__ == "__main__":
    # Get model name from environment
    model_name = os.environ.get("VLLM_MODEL", "facebook/opt-125m")

    # Start vLLM server
    cmd = ["vllm", "serve", model_name, "--host", "0.0.0.0", "--port", "8000"]
    subprocess.run(cmd)
EOF

RUN chmod +x /app/start_vllm.py

# Expose port
EXPOSE 8000

# Set entrypoint
CMD ["/app/start_vllm.py"]
"""

    def _generate_ollama_dockerfile(self, model_name: str) -> str:
        """Generate Dockerfile for Ollama-based image."""
        return f"""# Use Ubuntu base image
FROM ubuntu:22.04

# Set environment variables for Ollama
ENV PYTHONUNBUFFERED="1" \\
    RUNPOD_LLM_BACKEND="ollama" \\
    RUNPOD_LLM_MODEL_NAME="{model_name}" \\
    OLLAMA_MODEL="{model_name}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    git \\
    python3 \\
    python3-pip \\
    wget && \\
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Install Python dependencies
RUN pip3 install --root-user-action=ignore --no-cache-dir \\
    requests \\
    runpod \\
    transformers

# Create model directory and pre-download model
RUN mkdir -p /models
WORKDIR /models

# Pre-download the model during build
RUN /bin/bash -c '
model_name=$OLLAMA_MODEL
if [ -z "$model_name" ]; then
    model_name=llama2
fi
echo "Pre-downloading Ollama model: $model_name"

# Start Ollama server in background for model download
ollama serve &
OLLAMA_PID=$!

# Wait for server to start
sleep 5

# Pull the model
echo "Pulling model: $model_name"
if ollama pull "$model_name"; then
    echo "Successfully downloaded Ollama model: $model_name"
else
    echo "Warning: Could not pre-download Ollama model $model_name"
    echo "Model will be downloaded at runtime"
fi

# Stop the background server
kill $OLLAMA_PID 2>/dev/null || true
'

# Copy custom Ollama server script
COPY <<EOF /app/start_ollama.py
#!/usr/bin/env python3
import os
import sys
import subprocess
import time

if __name__ == "__main__":
    # Get model name from environment
    model_name = os.environ.get("OLLAMA_MODEL", "llama2")

    # Start Ollama server in background
    print("Starting Ollama server...")
    ollama_process = subprocess.Popen(["ollama", "serve"])

    # Wait for server to start
    time.sleep(5)

    # Ensure model is available (pull if needed)
    print(f"Ensuring model is available: {model_name}")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True, timeout=300)
        print(f"Model {model_name} is ready")
    except subprocess.TimeoutExpired:
        print(f"Warning: Model pull timed out, but continuing...")
    except subprocess.CalledProcessError:
        print(f"Warning: Could not pull model {model_name}, but continuing...")

    # Keep server running
    ollama_process.wait()
EOF

RUN chmod +x /app/start_ollama.py

# Expose port
EXPOSE 11434

# Set entrypoint
CMD ["/app/start_ollama.py"]
"""
