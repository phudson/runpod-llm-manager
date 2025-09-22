# 🧠 RunPod LLM Pod Manager with FastAPI Proxy

This advanced system automates the full lifecycle of GPU-backed LLM pods on RunPod, featuring enterprise-grade caching, monitoring, security, and a high-performance FastAPI reverse proxy for development tools like Continue, CodeGPT, and Prinova Cody.

## 🚀 Purpose

- 🚀 Launch ephemeral LLM pods with OpenAI-compatible endpoints
- 🔒 Default to **SECURE mode** for privacy and isolation
- ⚡ High-performance **FastAPI reverse proxy** with intelligent caching
- 📊 Real-time metrics and health monitoring
- 🔐 SSL/TLS support for secure communication
- 💾 LRU cache with configurable size limits
- 📈 Structured JSON logging and performance profiling
- 🏥 Comprehensive health checks and dashboard
- ⏰ Track pod state for restarts, shutdowns, and cost control
- 💰 Enforce runtime limits and prevent lingering charges
- 🤖 Support cron-based watchdog execution

## 📦 Files

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `manage_pod.py`               | 🚀 Unified lifecycle controller: start, restart, terminate, watchdog        |
| `proxy_fastapi.py`            | ⚡ FastAPI proxy with caching, metrics, SSL, and health monitoring          |
| `pod_config.json`             | ⚙️ Configuration file with model, GPU, cache, SSL, and runtime settings     |
| `pod_state.json`              | 📊 Auto-generated state file storing pod ID, model, and runtime info        |
| `.gitignore`                  | 🚫 Prevents committing sensitive files and local artifacts                 |
| `README.md`                   | 📖 This comprehensive documentation                                        |

## 🧰 Prerequisites & Installation

### 🧰 System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Debian, CentOS, etc.)
- macOS (10.15+)
- Windows with WSL2

**Python Version:**
- Python 3.8 or higher

### 📦 Installation

1. **Clone or download the repository:**
   ```bash
   git clone <repository-url>
   cd runpod-llm-manager
   ```

2. **Install Python dependencies:**
    ```bash
    pip install fastapi httpx uvicorn requests aiofiles
    ```

3. **Set up environment variables:**
   ```bash
   export RUNPOD_API_KEY="your-runpod-api-key-here"
   ```

4. **Create cache directory:**
   ```bash
   mkdir -p /tmp/llm_cache
   ```

> ✅ **No NGINX required!** The system uses a high-performance FastAPI proxy with built-in caching and monitoring.

## ⏱️ Cron Setup (Optional)

### 🐧 WSL2 / Ubuntu Setup Notes

To ensure `runpod-llm-manager` functions correctly inside WSL2 with Ubuntu:

#### ⚙️ Enable `systemd` (Optional but Recommended)

If you're using newer Ubuntu builds on WSL2, you can enable `systemd` for better service management:

1. Edit your WSL config:
   ```bash
   sudo nano /etc/wsl.conf
   ```
   Add:
   ```
   [boot]
   systemd=true
   ```

2. Restart WSL:
   ```powershell
   wsl.exe --shutdown
   ```

> Note: `systemd` support requires WSL version 0.67.6 or newer. Run `wsl --version` to check.

#### 🔧 Enable `cron` in WSL2

WSL2 does not start `cron` automatically. To enable it:

1. Install cron if not already present:
   ```bash
   sudo apt update
   sudo apt install cron
   ```

2. Allow passwordless startup for cron (optional but recommended for automation):
   ```bash
   sudo visudo
   ```
   Add this line at the bottom:
   ```
   your_username ALL=NOPASSWD: /usr/sbin/service cron start
   ```

3. Start cron manually once:
   ```bash
   sudo service cron start
   ```

4. To ensure cron starts automatically when WSL2 boots, use Windows Task Scheduler to run:
   ```bash
   wsl -d Ubuntu -- sudo service cron start
   ```
   on login or system boot.

To automate pod lifecycle management and prevent lingering charges, add the following cron entries:

### 🔄 Watchdog / Expiry Check (Every 5 Minutes)

Runs `manage_pod.py` to start, restart, or terminate pods based on runtime limits:

```cron
*/5 * * * * /usr/bin/python3 /path/to/runpod-llm-manager/manage_pod.py >> /var/log/runpod_watchdog.log 2>&1
```

### 🛑 Forced Termination (Midnight Daily)

Ensures all pods are terminated at midnight regardless of state:

```cron
0 0 * * * /usr/bin/python3 /path/to/runpod-llm-manager/manage_pod.py --shutdown >> /var/log/runpod_shutdown.log 2>&1
```

> Replace `/path/to/runpod-llm-manager/` with the actual path to your script.
> Ensure your user has permission to run Python without `sudo`.
> These cron jobs work with the existing code and provide automated lifecycle management.

## � Configuration

Create `pod_config.json` with comprehensive settings:

### 📋 Basic Configuration

```json
{
  "model": "deepseek-ai/deepseek-coder-33b-awq",
  "gpu_type_id": "NVIDIA RTX A6000",
  "runtime_seconds": 3600,
  "template_id": "vllm"
}
```

### ⚙️ Advanced Configuration

```json
{
  "model": "deepseek-ai/deepseek-coder-33b-awq",
  "gpu_type_id": "NVIDIA RTX A6000",
  "runtime_seconds": 3600,
  "template_id": "vllm",

  // Proxy Configuration
  "proxy_port": 8000,
  "cache_dir": "/tmp/llm_cache",

  // SSL/TLS Configuration
  "use_https": false,
  "ssl_cert": "/path/to/cert.pem",
  "ssl_key": "/path/to/key.pem",

  // Cache Configuration
  "max_cache_size": 1000,
  "max_cache_bytes": 1073741824,

  // Performance & Monitoring
  "enable_profiling": false
}
```

### 🔍 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Hugging Face model identifier |
| `gpu_type_id` | string | required | GPU type for pod deployment |
| `runtime_seconds` | int | 3600 | Maximum pod runtime in seconds |
| `template_id` | string | "vllm" | RunPod template identifier |
| `proxy_port` | int | 8000 | Local proxy port |
| `cache_dir` | string | "/tmp/llm_cache" | Cache directory path |
| `use_https` | boolean | false | Enable SSL/TLS |
| `ssl_cert` | string | null | SSL certificate file path |
| `ssl_key` | string | null | SSL private key file path |
| `max_cache_size` | int | 1000 | Maximum cached responses |
| `max_cache_bytes` | int | 1GB | Maximum cache size in bytes |
| `enable_profiling` | boolean | false | Enable debug/profiling endpoints |
| `initial_wait_seconds` | int | 10 | Seconds to wait after pod creation before checking status |
| `max_startup_attempts` | int | 20 | Maximum attempts to wait for pod to become ready |
| `poll_interval_seconds` | int | 5 | Seconds between pod status checks during startup |
### 🔍 Discovering Supported Models via RunPod UI

RunPod supports a wide range of open-source models for vLLM pods. To explore available options:

#### 🧭 Using Quick Deploy

1. Go to [RunPod Console](https://www.runpod.io/console)
2. Click **Deploy a Pod**
3. Select **Serverless > vLLM Worker**
4. In the **Model** dropdown, browse the list of supported Hugging Face models

These models are pre-tested for compatibility with RunPod’s vLLM container and expose an OpenAI-style API endpoint.

#### 📌 Notes

- Most models listed are public and do **not** require a Hugging Face token.
- If you select a gated model (e.g. `meta-llama/Llama-3-8B-Instruct`), you’ll need to provide a `HF_TOKEN` in your pod config.
- You can also deploy any compatible Hugging Face model manually by specifying its name in your `pod_config.json`.

> For examples of known working models, see the `models` list printed during `--refresh-catalog` in verbose mode.



## 🧪 Usage

### 🚀 Basic Usage

```bash
# Start or restart pod with watchdog behavior
python3 manage_pod.py

# Force termination of active pod
python3 manage_pod.py --shutdown

# Dry run mode (no actual API calls)
python3 manage_pod.py --dry-run

# Verbose logging
python3 manage_pod.py --verbose

# Refresh catalog and validate configuration
python3 manage_pod.py --refresh-catalog
```

### 🌐 API Endpoints

Once running, your LLM is available at:

```
http://localhost:8000/v1/chat/completions
```

#### 📊 Monitoring Endpoints

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics`
- **Dashboard**: `GET /dashboard`
- **Debug Cache** (if profiling enabled): `GET /debug/cache`

### 📈 Monitoring & Metrics

```bash
# Check proxy health
curl http://localhost:8000/health

# Get performance metrics
curl http://localhost:8000/metrics

# View comprehensive dashboard
curl http://localhost:8000/dashboard
```

### 🔧 Environment Variables

```bash
# Required
export RUNPOD_API_KEY="your-api-key"

# Optional (for advanced features)
export MAX_CACHE_SIZE="2000"          # Increase cache size
export CACHE_SIZE_BYTES="2147483648"  # 2GB cache
export ENABLE_PROFILING="true"        # Enable debug endpoints
export PREWARM_CACHE="true"           # Pre-populate cache with common patterns
```

## ✨ Advanced Features

### 💾 Intelligent Caching System

- **LRU Eviction**: Automatically removes least recently used cache entries
- **Size Management**: Configurable cache limits (entries and bytes)
- **SHA256 Hashing**: Fast, collision-resistant cache keys
- **Thread-Safe**: Concurrent access protection
- **Performance**: Sub-millisecond cache lookups

### 📊 Real-Time Monitoring

- **Health Checks**: Comprehensive system health monitoring
- **Performance Metrics**: Response times, cache hit rates, error rates
- **System Dashboard**: Complete system overview with configuration
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Debug Endpoints**: Cache inspection and profiling tools

### 🔐 Security & SSL/TLS

- **File Permissions**: Restricted access to sensitive files (0o600)
- **SSL/TLS Support**: HTTPS with configurable certificates
- **Environment Validation**: Early validation of API keys and configuration
- **Process Isolation**: Secure subprocess management

### ⚡ Performance Optimizations

- **Async Processing**: Non-blocking I/O with FastAPI and httpx
- **Connection Pooling**: Efficient HTTP client reuse
- **Graceful Shutdown**: 10-second timeout for clean process termination
- **Memory Management**: Controlled cache growth with eviction policies

### 🤖 Automation & Reliability

- **Lock Files**: Prevents concurrent execution conflicts
- **State Persistence**: Survives system restarts
- **Error Recovery**: Automatic pod restart on failures
- **Health Monitoring**: Continuous proxy health validation

## 🔐 Security Notes

- ✅ No privileged ports required
- ✅ No `sudo` needed for any operations
- ✅ File permissions automatically restricted
- ✅ SSL/TLS support for encrypted communication
- ✅ Environment variables for sensitive data
- ✅ Process isolation and secure cleanup

## VSCode extension configuration

### 🧩 Continue Extension

To connect the Continue extension to your locally hosted RunPod LLM endpoint, create or update the configuration file at:

```
~/.continue/config.json
```

with the following content:

```json
{
  "models": [
    {
      "title": "RunPod DeepSeek",
      "provider": "openai",
      "model": "deepseek-coder-33b-awq",
      "apiBase": "http://localhost:8080/v1"
    }
  ]
}
```

- **`title`**: Friendly display name for your model in Continue.
- **`provider`**: Must be `"openai"` since the RunPod endpoint is OpenAI-compatible.
- **`model`**: The exact model identifier you configured for your pod.
- **`apiBase`**: The local URL exposed by your FastAPI proxy (`localhost` and port should match your config, default: 8000).

This setup tells Continue to send requests to your RunPod pod’s OpenAI-compatible API endpoint running locally. Remember to restart the Continue extension after saving the config for changes to take effect.

### 🧩 VSCode CodeGPT Extension Configuration Example

To connect CodeGPT to your locally hosted RunPod LLM endpoint, open your VSCode `settings.json` file:

```bash
File → Preferences → Settings → Open Settings (JSON)
```

Add the following configuration:

```json
{
  "codegpt.model": "openai",
  "codegpt.apiKey": "sk-placeholder",
  "codegpt.apiBaseUrl": "http://localhost:8080/v1"
}
```

- **`model`**: Set to `"openai"` to use OpenAI-compatible formatting.
- **`apiKey`**: Required by CodeGPT even for local endpoints—use any placeholder string.
- **`apiBaseUrl`**: Must match your FastAPI proxy URL and port (default: `http://localhost:8000/v1`).

> ⚠️ CodeGPT requires a dummy API key even for local endpoints. You can use `"sk-local"` or `"sk-placeholder"`.

---

### 🧩 VSCode Prinova Cody Extension Configuration Example

Prinova Cody (Sourcegraph Cody) connects to LLMs via a Sourcegraph instance. To use a custom LLM like your RunPod pod, you'll need:

1. A Sourcegraph Enterprise instance
2. Admin access to configure external LLM endpoints
3. A generated access token

Once you have those:

- Open Cody in VSCode
- Click **Sign In to Your Enterprise Instance**
- Enter your Sourcegraph URL
- Paste your access token
- Select your custom model from the dropdown (if configured)

> ⚠️ Cody does not support direct local endpoint configuration in VSCode. You must register your RunPod endpoint with a Sourcegraph instance first.

For full setup instructions, see [Sourcegraph's Cody installation guide](https://sourcegraph.com/docs/cody/clients/install-vscode).

---

### 🧩 VSCode Kilo Code Extension Configuration

To connect the Kilo Code extension to your locally hosted RunPod LLM proxy:

1. Install the **Kilo Code** extension in VSCode
2. Go to VSCode Settings → Extensions → Kilo Code
3. Configure the following basic settings:

```json
{
  "kilo-code.api.baseUrl": "http://localhost:8000/v1",
  "kilo-code.api.key": "sk-local-proxy",
  "kilo-code.model.name": "deepseek-coder-33b-awq",
  "kilo-code.cache.enabled": true,
  "kilo-code.cache.directory": "/tmp/llm_cache"
}
```

#### Testing the Configuration

1. **Start the proxy:**
    ```bash
    python3 manage_pod.py
    ```

2. **Verify proxy health:**
    ```bash
    curl http://localhost:8000/health
    ```

3. **Test in VSCode:**
    - Open a Python file in VSCode
    - Use Kilo Code autocomplete or chat features
    - Verify requests are routed through your local proxy

#### Troubleshooting

- **Connection issues**: Ensure proxy is running on port 8000
- **Authentication errors**: Verify the API key matches `"sk-local-proxy"`
- **Slow responses**: Check cache is working with `curl http://localhost:8000/metrics`

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### ❌ `Import "uvicorn" could not be resolved`
**Solution**: Install missing dependencies:
```bash
pip install uvicorn
```

#### ❌ `Permission denied` when creating cache files
**Solution**: Ensure cache directory permissions:
```bash
mkdir -p /tmp/llm_cache
chmod 755 /tmp/llm_cache
```

#### ❌ `Port already in use`
**Solution**: Change proxy port in `pod_config.json`:
```json
{
  "proxy_port": 8001
}
```

#### ❌ SSL certificate errors
**Solution**: Verify certificate files exist and have correct permissions:
```bash
ls -la /path/to/cert.pem /path/to/key.pem
chmod 600 /path/to/key.pem
```

#### ❌ Pod fails health check
**Solution**: Check pod status and logs:
```bash
python3 manage_pod.py --verbose
```

#### ❌ Cache not working
**Solution**: Check cache directory and permissions:
```bash
ls -la /tmp/llm_cache/
```

### 📊 Monitoring Commands

```bash
# Check if proxy is running
ps aux | grep proxy_fastapi

# Check proxy health
curl http://localhost:8000/health

# View cache statistics
curl http://localhost:8000/metrics

# View comprehensive dashboard
curl http://localhost:8000/dashboard
```

### 🔍 Debug Mode

Enable debug endpoints for troubleshooting:

```json
{
  "enable_profiling": true
}
```

Then access debug information:
```bash
curl http://localhost:8000/debug/cache
```

## 🧹 Cleanup

### Manual Cleanup
```bash
# Stop the proxy and terminate pod
python3 manage_pod.py --shutdown

# Clean up cache files (optional)
rm -rf /tmp/llm_cache/*

# Remove state files
rm -f pod_state.json
rm -f /tmp/fastapi_proxy.pid
```

### Automated Cleanup
The system automatically:
- ✅ Terminates pods on expiry
- ✅ Cleans up PID files on shutdown
- ✅ Removes stale lock files
- ✅ Evicts old cache entries

### Cache Management
```bash
# View cache size
du -sh /tmp/llm_cache/

# Clear all cache
rm -rf /tmp/llm_cache/*
mkdir -p /tmp/llm_cache
```

---

## ⚡ Performance Optimization

### Cache Performance Tips

- **Use fast storage**: Place cache directory on SSD/NVMe for better performance
- **Pre-warm cache**: Enable `PREWARM_CACHE=true` to populate cache with common patterns on startup
- **Monitor cache hit rates**: Use `/metrics` endpoint to track cache effectiveness
- **Adjust cache size**: Increase `MAX_CACHE_SIZE` for better hit rates on large codebases

### SSL/TLS Configuration
```json
{
  "use_https": true,
  "ssl_cert": "/path/to/cert.pem",
  "ssl_key": "/path/to/key.pem"
}
```

---

## 🤝 Contributing

This system is designed to be:
- **Modular**: Easy to extend with new features
- **Configurable**: All major settings are configurable
- **Observable**: Comprehensive logging and metrics
- **Secure**: Follows security best practices

### Adding New Features
1. Add configuration options to `pod_config.json`
2. Implement functionality in appropriate module
3. Add metrics and logging
4. Update documentation
5. Test thoroughly

---

## 📄 License

This project is open source. Please ensure compliance with:
- RunPod Terms of Service
- Hugging Face model licenses
- Local data privacy regulations

---

## 🆘 Support

### Getting Help
1. Check the troubleshooting section above
2. Check proxy health: `curl http://localhost:8000/health`
3. Enable verbose mode: `python3 manage_pod.py --verbose`
4. Check pod status: `python3 manage_pod.py --refresh-catalog`

### Common Resources
- [RunPod Documentation](https://docs.runpod.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Models](https://huggingface.co/models)

---

*Last updated: 2025-09-22*
