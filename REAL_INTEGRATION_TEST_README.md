# Real Integration Test for RunPod LLM Manager

This document explains how to run the real integration test that builds Docker images, pushes them to a registry, creates RunPod pods, and tests LLM functionality.

## Prerequisites

### 1. Docker Installation
Ensure Docker is installed and running on your system:
```bash
docker --version
```

### 2. Docker Registry Access
You need access to push images to a Docker registry. The test uses Docker Hub by default.

#### Option A: Docker Hub (Recommended)
1. Create a Docker Hub account at https://hub.docker.com/
2. Create a Personal Access Token (PAT):
   - Go to Account Settings > Security > Personal Access Tokens
   - Click "Generate new token"
   - Give it a descriptive name (e.g., "RunPod Integration Test")
   - Select appropriate permissions (at minimum: Read, Write, Delete)
   - Copy the token immediately (you won't see it again)

#### Option B: Private Registry
Modify the `DockerImageBuilderService` to use your private registry.

### 3. RunPod Account and API Key
1. Create a RunPod account at https://runpod.io/
2. Generate an API key:
   - Go to Account Settings > API Keys
   - Create a new API key
   - Copy the API key

## Required Environment Variables

Set these environment variables before running the test:

```bash
# Docker Registry Credentials (Personal Access Token)
export DOCKER_USERNAME="your_dockerhub_username"
export DOCKER_TOKEN="your_dockerhub_personal_access_token"

# RunPod API Credentials
export RUNPOD_API_KEY="your_runpod_api_key"

# Optional: Custom RunPod endpoint (defaults to https://api.runpod.ai)
export RUNPOD_ENDPOINT="https://api.runpod.ai"
```

## Running the Test

### Step 1: Set Environment Variables
```bash
export DOCKER_USERNAME="your_dockerhub_username"
export DOCKER_TOKEN="your_dockerhub_personal_access_token"
export RUNPOD_API_KEY="your_runpod_api_key"
```

### Step 2: Run the Integration Test
```bash
python test_real_docker_integration.py
```

## What the Test Does

The integration test performs these steps:

1. **Environment Check**: Validates all required credentials and Docker installation
2. **Registry Login**: Logs into Docker Hub using provided credentials
3. **Image Build**: Builds a custom Docker image for a small LLM model (facebook/opt-125m)
4. **Image Push**: Pushes the built image to Docker Hub
5. **Pod Creation**: Creates a RunPod pod using the custom image
6. **Pod Monitoring**: Waits for the pod to become ready (up to 5 minutes)
7. **LLM Testing**: Sends a test prompt to the LLM running in the pod
8. **Validation**: Verifies the LLM response structure and content
9. **Cleanup**: Terminates the created pod (images remain for reuse)

## Expected Output

```
2025-10-09 16:55:00,000 - INFO - Starting real Docker integration test...
2025-10-09 16:55:00,001 - INFO - Docker available: Docker version 24.0.6, build ed223bc
2025-10-09 16:55:00,002 - INFO - Logging into Docker registry...
2025-10-09 16:55:00,003 - INFO - Successfully logged into Docker registry
2025-10-09 16:55:00,004 - INFO - Building and pushing image for model: facebook/opt-125m, backend: vllm
2025-10-09 16:55:00,005 - INFO - Created Dockerfile at: /tmp/tmpXXX/Dockerfile
2025-10-09 16:55:00,006 - INFO - Building Docker image: runpod/llm-images:vllm-facebook-opt-125m-abc123
2025-10-09 16:55:02,000 - INFO - Successfully built Docker image
2025-10-09 16:55:02,001 - INFO - Pushing Docker image: runpod/llm-images:vllm-facebook-opt-125m-abc123
2025-10-09 16:55:05,000 - INFO - Successfully pushed Docker image
2025-10-09 16:55:05,001 - INFO - Creating pod with image: runpod/llm-images:vllm-facebook-opt-125m-abc123
2025-10-09 16:55:05,002 - INFO - Created pod: abc123def456
2025-10-09 16:55:05,003 - INFO - Waiting for pod to be ready...
2025-10-09 16:55:15,000 - INFO - Pod status (attempt 1/30): CREATED
2025-10-09 16:55:25,000 - INFO - Pod status (attempt 2/30): RUNNING
2025-10-09 16:55:25,001 - INFO - Pod is ready!
2025-10-09 16:55:25,002 - INFO - Testing LLM functionality on pod...
2025-10-09 16:55:25,003 - INFO - LLM Response received:
2025-10-09 16:55:25,004 - INFO -   Model: facebook/opt-125m
2025-10-09 16:55:25,005 - INFO -   Choices: 1
2025-10-09 16:55:25,006 - INFO -   Content: Hello! I'm a helpful AI assistant...
2025-10-09 16:55:25,007 - INFO - LLM functionality test passed!
2025-10-09 16:55:25,008 - INFO - 🎉 Integration test completed successfully!
2025-10-09 16:55:25,009 - INFO -    - Built and pushed image: runpod/llm-images:vllm-facebook-opt-125m-abc123
2025-10-09 16:55:25,010 - INFO -    - Created and tested pod: abc123def456
2025-10-09 16:55:25,011 - INFO - Cleaning up resources...
2025-10-09 16:55:25,012 - INFO - Terminating pod: abc123def456
2025-10-09 16:55:25,013 - INFO - Successfully terminated pod: abc123def456
```

## Troubleshooting

### Docker Login Issues
- Ensure your Docker Hub username and personal access token are correct
- Make sure your Personal Access Token has the required permissions (Read, Write, Delete)
- Personal Access Tokens don't expire but can be revoked
- Try manual login first: `docker login -u $DOCKER_USERNAME -p $DOCKER_TOKEN`

### Image Build Failures
- Check Docker has sufficient disk space
- Ensure internet connectivity for downloading base images
- Check the Dockerfile content in the temporary directory

### Pod Creation Issues
- Verify your RunPod API key is valid and has sufficient credits
- Check that the GPU type "NVIDIA GeForce RTX 3090" is available in your region
- Ensure your RunPod account has pod creation permissions

### LLM Testing Failures
- The pod might take longer than 5 minutes to start - increase `max_attempts`
- Check that the model (facebook/opt-125m) is compatible with vLLM
- Verify the pod's container logs if the LLM doesn't respond

## Cost Considerations

This test will incur costs on RunPod:
- Pod runtime costs (typically $0.10-0.50/hour depending on GPU type)
- The test pod runs for 5-10 minutes maximum
- Images are cached in Docker Hub for future reuse

Monitor your RunPod account balance and stop the test if needed.

## Security Notes

- Never commit API keys, personal access tokens, or passwords to version control
- Use Docker Personal Access Tokens instead of passwords for enhanced security
- Personal Access Tokens can be scoped to specific permissions and revoked individually
- The test creates resources with unique names to avoid conflicts
- All created pods are terminated at the end of the test

## Customization

You can modify the test parameters in the `run_test()` method:

```python
# Change the model
model_name = "microsoft/DialoGPT-small"

# Change the backend
backend = "ollama"

# Change GPU type
pod_config["gpu_type_id"] = "NVIDIA A100"

# Change test prompt
request = ChatCompletionRequest(
    model=model_name,
    messages=[ChatMessage(role="user", content="Your custom prompt here")],
    max_tokens=100,
    temperature=0.8,
    stream=False
)
