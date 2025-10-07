# Test Strategy for RunPod LLM Pod Manager with FastAPI Proxy

## Overview

This test strategy covers comprehensive testing of the RunPod LLM Pod Manager system with Model Store integration, including both manual and automated components. The strategy focuses on validating the full lifecycle from manual pod setup through automated management, with emphasis on VSCode integration for development tools like Kilo Code and proper Model Store model selection.

## Test Objectives

- ✅ Validate manual pod creation and proxy configuration
- ✅ Test FastAPI proxy functionality and caching
- ✅ Verify VSCode extension integration (Kilo Code)
- ✅ Test automated pod management via manage_pod.py
- ✅ Validate Model Store integration and model selection
- ✅ Ensure proper cleanup and cost control
- ✅ Validate performance optimizations and monitoring

## Prerequisites

### System Requirements
- Linux/macOS/Windows with Python 3.8+
- RunPod API key with pod creation permissions
- VSCode with Kilo Code extension installed
- Internet connection for RunPod API access

### Environment Setup
```bash
# Install dependencies
pip install fastapi httpx uvicorn requests aiofiles

# Set environment variables
export RUNPOD_API_KEY="your-runpod-api-key-here"

# Create cache directory
mkdir -p /tmp/llm_cache
```

### Test Data Preparation
- Prepare test model configurations (deepseek, mistral, etc.)
- Set up test cache directory with appropriate permissions
- Ensure VSCode is configured for extension testing

---

## Phase 1: Manual Pod Setup and Proxy Testing

### 1.1 Manual Pod Creation on RunPod

**Objective**: Create a pod manually via RunPod UI to test basic connectivity.

**Steps**:
1. Navigate to [RunPod Console](https://www.runpod.io/console)
2. Click "Deploy a Pod"
3. Select "Serverless > vLLM Worker"
4. Configure pod settings:
   - **Template**: vLLM (latest)
   - **GPU Type**: NVIDIA RTX A6000 (or available GPU)
   - **Model**: deepseek-ai/deepseek-coder-33b-awq
   - **Cloud Type**: SECURE
   - **Container Disk**: 20GB
   - **Volume**: 0GB
   - **Start Jupyter**: Disabled
5. Configure Model Store:
   - **Model**: Select from available Model Store models (e.g., deepseek-ai/deepseek-coder-33b-awq)
6. Click "Deploy" and wait for pod to become RUNNING
7. Note the pod's **Public IP** and **Port** (typically 8000)

**Expected Results**:
- Pod status shows "RUNNING"
- Public IP and port are accessible
- No errors in RunPod console

### 1.2 Manual Proxy Startup

**Objective**: Start proxy_fastapi.py manually with environment variables pointing to the manual pod.

**Steps**:
1. Set environment variables for the proxy:
   ```bash
   export RUNPOD_ENDPOINT="http://[POD_IP]:[POD_PORT]/v1/chat/completions"
   export CACHE_DIR="/tmp/llm_cache"
   export MAX_CACHE_SIZE="1000"
   export CACHE_SIZE_BYTES="1073741824"  # 1GB
   ```

2. Start the FastAPI proxy:
   ```bash
   python3 proxy_fastapi.py
   ```

3. Verify proxy startup:
   ```bash
   # Check if proxy is running
   curl http://localhost:8000/health

   # Expected response:
   {
     "status": "healthy",
     "endpoint": "http://[POD_IP]:[POD_PORT]/v1/chat/completions",
     "cache_dir": "/tmp/llm_cache",
     "uptime": 5.2
   }
   ```

**Expected Results**:
- Proxy starts without errors
- Health endpoint returns 200 OK
- Logs show successful connection to backend

### 1.3 Basic Proxy Functionality Testing

**Objective**: Test core proxy features before VSCode integration.

**Steps**:
1. Test basic chat completion:
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "deepseek-coder-33b-awq",
       "messages": [{"role": "user", "content": "Write a hello world function in Python"}],
       "max_tokens": 100
     }'
   ```

2. Test caching (repeat the same request):
   ```bash
   # First request (cache miss)
   time curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "def hello"}], "max_tokens": 50}'

   # Second request (cache hit - should be faster)
   time curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "def hello"}], "max_tokens": 50}'
   ```

3. Check metrics:
   ```bash
   curl http://localhost:8000/metrics

   # Expected: cache_hits > 0, cache_misses > 0
   ```

**Expected Results**:
- First request: 2-5 seconds (backend processing)
- Second request: <0.5 seconds (cache hit)
- Metrics show proper cache statistics

---

## Phase 2: VSCode Configuration and Testing

### 2.1 Kilo Code Extension Configuration

**Objective**: Configure VSCode Kilo Code extension to use the local proxy.

**Steps**:
1. Open VSCode Settings (Ctrl/Cmd + ,)
2. Search for "Kilo Code" or navigate to Extensions > Kilo Code
3. Configure the following settings:

   **Method 1: Extension Settings**
   ```json
   {
     "kilo-code.api.baseUrl": "http://localhost:8000/v1",
     "kilo-code.api.key": "sk-local-proxy",
     "kilo-code.model.name": "deepseek-coder-33b-awq",
     "kilo-code.cache.enabled": true,
     "kilo-code.cache.directory": "/tmp/llm_cache"
   }
   ```

   **Method 2: settings.json**
   ```json
   {
     "kilo-code": {
       "api": {
         "baseUrl": "http://localhost:8000/v1",
         "key": "sk-local-proxy"
       },
       "model": {
         "name": "deepseek-coder-33b-awq"
       },
       "cache": {
         "enabled": true,
         "directory": "/tmp/llm_cache"
       }
     }
   }
   ```

4. Restart VSCode to apply changes

**Expected Results**:
- VSCode settings saved without errors
- Kilo Code extension shows connected status

### 2.2 VSCode Integration Testing

**Objective**: Test Kilo Code functionality with the local proxy.

**Steps**:
1. Open a Python file in VSCode
2. Test autocomplete:
   - Type `def ` and wait for suggestions
   - Verify suggestions come from the model

3. Test chat functionality:
   - Open Kilo Code chat panel
   - Ask a coding question: "Write a function to reverse a string"
   - Verify response appears and is cached

4. Test code generation:
   - Highlight code and use "Generate similar code"
   - Verify generated code quality

5. Monitor proxy logs during testing:
   ```bash
   # In another terminal, watch proxy logs
   tail -f /tmp/proxy_fastapi.log
   ```

6. Check cache growth:
   ```bash
   # Monitor cache directory
   watch -n 5 'ls -la /tmp/llm_cache/ | wc -l'

   # Check cache metrics
   curl -s http://localhost:8000/metrics | jq ".cache_entries"
   ```

**Expected Results**:
- Autocomplete suggestions appear within 1-2 seconds
- Chat responses are coherent and relevant
- Cache entries increase with repeated requests
- No errors in VSCode or proxy logs

### 2.3 Performance Testing in VSCode

**Objective**: Validate performance optimizations for development workflow.

**Steps**:
1. Test response times for common coding patterns:
   ```bash
   # Test various code patterns
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "class "}], "max_tokens": 50}'

   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "import "}], "max_tokens": 30}'
   ```

2. Test cache hit rates:
   ```bash
   # Repeat requests to build cache
   for i in {1..5}; do
     curl -s -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "def factorial"}], "max_tokens": 100}' > /dev/null
   done

   # Check hit rate
   curl -s http://localhost:8000/metrics | jq ".cache_hit_rate"
   ```

3. Test concurrent requests (simulate multiple VSCode instances):
   ```bash
   # Run multiple requests in parallel
   for i in {1..3}; do
     curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "print('\"'test $i'\"')"}], "max_tokens": 20}' &
   done
   wait
   ```

**Expected Results**:
- Cache hit rate > 80% after repeated requests
- Concurrent requests handled without errors
- Response times improve with caching

---

## Phase 3: Managed Pod Testing

### 3.1 Model Store Validation

**Objective**: Verify Model Store integration and available models.

**Steps**:
1. Test catalog refresh with Model Store:
   ```bash
   python3 manage_pod.py --refresh-catalog --verbose
   ```

2. Verify Model Store models are listed:
   - Check that modelStoreIds are displayed
   - Confirm fallback models work if Model Store unavailable

3. Validate modelStoreId configuration:
   - Ensure modelStoreId in pod_config.json matches available models
   - Test with invalid modelStoreId (should fail validation)

**Expected Results**:
- Model Store models fetched successfully
- Verbose output shows modelStoreIds and names
- Configuration validation passes with valid modelStoreId

### 3.2 Configuration Setup

**Objective**: Set up pod_config.json for automated testing.

**Steps**:
1. Update pod_config.json:
   ```json
   {
     "modelStoreId": "deepseek-ai/deepseek-coder-33b-awq",
     "gpu_type_id": "NVIDIA RTX A6000",
     "runtime_seconds": 1800,
     "template_id": "vllm",
     "proxy_port": 8000,
     "cache_dir": "/tmp/llm_cache",
     "max_cache_size": 1000,
     "max_cache_bytes": 1073741824,
     "enable_profiling": true
   }
   ```

2. Ensure RUNPOD_API_KEY is set:
   ```bash
   echo $RUNPOD_API_KEY
   ```

**Expected Results**:
- Configuration file is valid JSON
- API key is properly set

### 3.2 Automated Pod Startup

**Objective**: Test the full automated pod lifecycle.

**Steps**:
1. Stop manual proxy if running:
   ```bash
   pkill -f proxy_fastapi.py
   ```

2. Start managed pod:
   ```bash
   python3 manage_pod.py --verbose
   ```

3. Monitor startup process:
   ```bash
   # Watch logs for pod creation
   tail -f /var/log/runpod_watchdog.log 2>/dev/null || echo "No log file yet"

   # Check pod status
   python3 manage_pod.py --verbose
   ```

4. Verify proxy is running:
   ```bash
   curl http://localhost:8000/health
   ```

**Expected Results**:
- Pod created successfully via API
- Proxy starts automatically
- Health check passes
- State file created: pod_state.json

### 3.3 Managed Pod Functionality Testing

**Objective**: Test all managed features.

**Steps**:
1. Test basic functionality:
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-coder-33b-awq", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
   ```

2. Test state persistence:
   ```bash
   # Check state file
   cat pod_state.json

   # Restart manage_pod.py (should detect existing pod)
   python3 manage_pod.py --verbose
   ```

3. Test expiry handling:
   ```bash
   # Wait for pod to expire or manually test shutdown
   python3 manage_pod.py --shutdown
   ```

4. Test error recovery:
   - Simulate network issues
   - Test pod restart scenarios
   - Verify lock file handling

**Expected Results**:
- Pod state persists across restarts
- Automatic cleanup on expiry
- Proper error handling and recovery

### 3.4 VSCode Testing with Managed Pod

**Objective**: Ensure VSCode integration works with managed setup.

**Steps**:
1. Verify VSCode configuration is still correct
2. Test Kilo Code functionality with managed pod
3. Compare performance between manual and managed setups
4. Test cache persistence across pod restarts

**Expected Results**:
- No configuration changes needed
- Same functionality as manual setup
- Cache preserved across restarts

---

## Phase 4: Cleanup and Verification

### 4.1 Manual Cleanup

**Steps**:
1. Stop managed pod:
   ```bash
   python3 manage_pod.py --shutdown
   ```

2. Clean up files:
   ```bash
   rm -f pod_state.json
   rm -f /tmp/fastapi_proxy.pid
   rm -rf /tmp/llm_cache/*
   ```

3. Terminate manual pod via RunPod UI

**Expected Results**:
- All processes stopped
- No lingering files or processes
- RunPod console shows pod terminated

### 4.2 Verification Steps

**Steps**:
1. Check for running processes:
   ```bash
   ps aux | grep -E "(manage_pod|proxy_fastapi)"
   ```

2. Verify no open ports:
   ```bash
   netstat -tlnp | grep :8000
   ```

3. Check RunPod console for any active pods

**Expected Results**:
- No related processes running
- Port 8000 free
- No active pods in RunPod account

---

## Test Metrics and Success Criteria

### Performance Benchmarks
- **Cache Hit Rate**: >80% for repeated requests
- **Response Time**: <2 seconds for cache hits, <5 seconds for misses
- **Startup Time**: <60 seconds for pod creation
- **Memory Usage**: <2GB for proxy with 1000 cache entries

### Functional Success Criteria
- ✅ Manual pod creation and proxy startup
- ✅ VSCode extension configuration and functionality
- ✅ Automated pod management and lifecycle
- ✅ Model Store integration and model validation
- ✅ Proper cleanup and cost control
- ✅ Error handling and recovery
- ✅ Cache functionality and performance

### Monitoring Commands
```bash
# Health check
curl http://localhost:8000/health

# Performance metrics
curl http://localhost:8000/metrics

# Cache inspection
curl http://localhost:8000/debug/cache

# Pod status
python3 manage_pod.py --verbose
```

---

## Risk Mitigation

### Common Issues and Solutions

1. **API Key Issues**:
   - Verify RUNPOD_API_KEY is set correctly
   - Check API key permissions for pod creation

2. **Network Connectivity**:
   - Ensure stable internet connection
   - Test RunPod API accessibility

3. **Port Conflicts**:
   - Verify port 8000 is available
   - Change proxy_port in configuration if needed

4. **Cache Permission Issues**:
   - Ensure /tmp/llm_cache has write permissions
   - Check disk space availability

5. **VSCode Extension Issues**:
   - Restart VSCode after configuration changes
   - Verify extension is installed and enabled

### Emergency Procedures

1. **Force Shutdown**:
   ```bash
   python3 manage_pod.py --shutdown
   pkill -f proxy_fastapi.py
   ```

2. **Manual Pod Termination**:
   - Use RunPod UI to terminate pods
   - Check for any remaining charges

3. **Cache Reset**:
   ```bash
   rm -rf /tmp/llm_cache/*
   mkdir -p /tmp/llm_cache
   ```

---

## Test Environment Setup

### Development Environment
- Python 3.8+ with required packages
- VSCode with Kilo Code extension
- RunPod account with credits
- Stable internet connection

### Test Data
- Standard coding prompts for cache testing
- Various model configurations
- Different GPU types for performance comparison

### Logging and Monitoring
- Enable verbose logging during testing
- Monitor RunPod console for pod status
- Track API usage and costs
- Log all test results and observations

This comprehensive test strategy ensures thorough validation of both manual and automated components of the RunPod LLM Pod Manager system, with particular emphasis on VSCode integration and development workflow optimization.
