#!/usr/bin/env python3
"""
RunPod LLM Manager - Service Layer Architecture
Modern CLI tool for managing RunPod pods with Model Store integration
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests

from .config import config
from .dependencies import get_dependencies
from .pod_service import PodManagementService
from .serverless_service import ServerlessService

# CLI flags
verbose = "--verbose" in sys.argv
dry_run = "--dry-run" in sys.argv
refresh_flag = "--refresh-catalog" in sys.argv


def log(msg):
    """Log message if verbose mode is enabled."""
    if verbose:
        print(f"[LOG] {msg}")


# Constants
CONFIG_PATH = os.getenv("CONFIG_PATH", "llm_config.json")
STATE_PATH = os.getenv("STATE_PATH", "pod_state.json")
PROXY_PID_FILE = os.getenv("PROXY_PID_FILE", "/tmp/fastapi_proxy.pid")
LOCKFILE = os.getenv("LOCKFILE", "/tmp/runpod_manage.lock")


def get_api_key():
    """Get RunPod API key from environment."""
    key = os.getenv("RUNPOD_API_KEY")
    if not key or not re.match(r"^[a-zA-Z0-9_\-]{32,}$", key):
        print("❌ Invalid or missing RUNPOD_API_KEY.")
        sys.exit(1)
    return key


def validate_environment():
    """Validate required environment variables."""
    if not os.getenv("RUNPOD_API_KEY"):
        print("❌ RUNPOD_API_KEY environment variable not set")
        print("Set it with: export RUNPOD_API_KEY='your-api-key'")
        sys.exit(1)


def acquire_lock():
    """Acquire file-based lock to prevent concurrent execution."""
    if os.path.exists(LOCKFILE):
        with open(LOCKFILE) as f:
            pid = f.read().strip()
        if pid and pid.isdigit():
            try:
                os.kill(int(pid), 0)
                print(f"⛔ Another instance is running (PID {pid}). Exiting.")
                sys.exit(1)
            except OSError:
                log("⚠️ Stale lockfile detected. Proceeding.")
        else:
            log("⚠️ Invalid lockfile contents. Proceeding.")
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))
    log("🔒 Lockfile acquired.")


def release_lock():
    """Release the file-based lock."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)
        log("🔓 Lockfile released.")


def load_config():
    """Load pod configuration from file."""
    log(f"Loading {CONFIG_PATH}...")
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_state(state):
    """Save pod state to file."""
    log(f"Saving {STATE_PATH}...")
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Load pod state from file."""
    if not os.path.exists(STATE_PATH):
        log("No existing pod state found.")
        return None
    log(f"Loading {STATE_PATH}...")
    with open(STATE_PATH) as f:
        return json.load(f)


def pod_is_expired(state, config):
    """Check if pod has expired based on runtime limits."""
    start = datetime.fromisoformat(state["start_time"])
    end = datetime.fromisoformat(state["end_time"])
    expired = datetime.now(timezone.utc) > end
    log(f"Pod started at {start}, expires at {end}. Expired: {expired}")
    return expired


async def shutdown_async(state):
    """Shutdown active pod/endpoint and proxy using service layer."""
    # Stop the proxy first
    stop_proxy()

    mode = state.get("mode", "pod")
    deps = get_dependencies()

    if mode == "pod":
        pod_id = state.get("pod_id")
        if pod_id:
            log(f"Terminating pod {pod_id}...")
            pod_service = PodManagementService(deps)
            await pod_service.terminate_pod(pod_id)
            if os.path.exists(STATE_PATH):
                os.remove(STATE_PATH)
            log("Pod state file removed.")
            print(f"🛑 Pod {pod_id} terminated.")
        else:
            print("ℹ️ No active pod to shut down.")
    elif mode == "serverless":
        endpoint_id = state.get("endpoint_id")
        if endpoint_id:
            log(f"Deleting endpoint {endpoint_id}...")
            serverless_service = ServerlessService(deps)
            await serverless_service.delete_endpoint(endpoint_id)
            if os.path.exists(STATE_PATH):
                os.remove(STATE_PATH)
            log("Endpoint state file removed.")
            print(f"🛑 Endpoint {endpoint_id} deleted.")
        else:
            print("ℹ️ No active endpoint to shut down.")


def shutdown(state):
    """Shutdown active pod/endpoint and proxy."""
    import asyncio

    asyncio.run(shutdown_async(state))


async def refresh_catalog_async():
    """Refresh and display RunPod catalog information using service layer."""
    log("Refreshing RunPod catalog...")
    deps = get_dependencies()
    pod_service = PodManagementService(deps)
    serverless_service = ServerlessService(deps)

    catalog: Dict[str, Any] = {}

    # Get available templates from API
    templates_response = await deps.http_client.get(
        "https://api.runpod.ai/v2/templates",
        headers={"Authorization": f"Bearer {deps.config.runpod_api_key}"},
    )

    if not isinstance(templates_response, list):
        raise ValueError(
            f"Invalid API response for templates: expected list, got {type(templates_response)}"
        )

    if not templates_response:
        raise ValueError("No templates available from RunPod API")

    catalog["templates"] = []
    for template in templates_response:
        if isinstance(template, dict):
            template_info = {
                "id": template.get("id", ""),
                "name": template.get("name", ""),
                "image_name": template.get("imageName", ""),
                "description": template.get("description", ""),
                "gpu_types": template.get("gpuTypes", []),
                "is_vllm": "vllm" in template.get("name", "").lower()
                or "llm" in template.get("name", "").lower(),
            }
            catalog["templates"].append(template_info)

    # Get GPU types (hardcoded for now as API may not provide this)
    catalog["gpu_types"] = [
        {"id": "NVIDIA RTX A6000", "name": "NVIDIA RTX A6000", "vram_gb": 48},
        {"id": "NVIDIA RTX 4090", "name": "NVIDIA RTX 4090", "vram_gb": 24},
        {"id": "NVIDIA A100 80GB", "name": "NVIDIA A100 80GB", "vram_gb": 80},
    ]

    # Get model store information (would need real ModelStore API)
    catalog["model_store"] = []
    known_models = [
        "deepseek-ai/deepseek-coder-33b-awq",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/DialoGPT-large",
        "facebook/opt-125m",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
    ]

    for model_id in known_models:
        model_info = {
            "id": model_id,
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "cached": True,  # Assume cached for demo
            "size_gb": 2.5,  # Placeholder
        }
        catalog["model_store"].append(model_info)

    log(f"Successfully fetched catalog with {len(catalog['templates'])} templates")
    return catalog


def refresh_catalog():
    """Refresh and display RunPod catalog information."""
    try:
        catalog = asyncio.run(refresh_catalog_async())
    except Exception as e:
        print(f"❌ Failed to refresh catalog: {e}")
        print("This may be due to network issues or invalid API credentials.")
        print("Check your RUNPOD_API_KEY and internet connection.")
        sys.exit(1)

    if verbose:
        print("📦 RunPod Catalog:")

        # Display templates
        if "templates" in catalog and catalog["templates"]:
            print("Templates:")
            for template in catalog["templates"][:5]:  # Show first 5
                vllm_indicator = " (vLLM)" if template.get("is_vllm") else ""
                print(f"  • {template['id']} - {template['name']}{vllm_indicator}")

        # Display GPU types
        if "gpu_types" in catalog and catalog["gpu_types"]:
            print("GPU Types:")
            for gpu in catalog["gpu_types"]:
                print(f"  • {gpu['id']} ({gpu['vram_gb']}GB VRAM)")

        # Display model store
        if "model_store" in catalog and catalog["model_store"]:
            print("Model Store:")
            for model in catalog["model_store"][:5]:  # Show first 5
                cached_indicator = " (cached)" if model.get("cached") else ""
                print(f"  • {model['id']}{cached_indicator}")

    return catalog


def validate_config(config, catalog):
    """Validate configuration against catalog."""
    errors = []

    mode = config.get("mode", "pod")
    if mode not in ["pod", "serverless"]:
        errors.append(f"Invalid mode: {mode}. Must be 'pod' or 'serverless'")

    # Validate model configuration
    if "model" not in config:
        errors.append("Missing 'model' configuration section")
    else:
        model_config = config["model"]
        if "name" not in model_config:
            errors.append("Missing 'model.name' in configuration")

    # Validate compute configuration
    if "compute" not in config:
        errors.append("Missing 'compute' configuration section")
    else:
        compute_config = config["compute"]
        if "gpu_type_id" not in compute_config:
            errors.append("Missing 'compute.gpu_type_id' in configuration")

    if mode == "pod":
        # Validate pod-specific configuration
        if "pod" not in config:
            errors.append("Missing 'pod' configuration section for pod mode")
        else:
            pod_config = config["pod"]
            if "template_id" not in pod_config:
                errors.append("Missing 'pod.template_id' in configuration")

    elif mode == "serverless":
        # Validate serverless-specific configuration
        if "serverless" not in config:
            errors.append("Missing 'serverless' configuration section for serverless mode")
        else:
            serverless_config = config["serverless"]
            if "template_id" not in serverless_config:
                errors.append("Missing 'serverless.template_id' in configuration")

    if errors:
        print("❌ Configuration validation failed:")
        for e in errors:
            print("-", e)
        sys.exit(1)
    else:
        log("✅ Configuration validated successfully.")


def is_proxy_running():
    """Check if the FastAPI proxy is still running."""
    if os.path.exists(PROXY_PID_FILE):
        try:
            with open(PROXY_PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # Check if process exists
            return True
        except (OSError, ValueError):
            if os.path.exists(PROXY_PID_FILE):
                os.remove(PROXY_PID_FILE)
    return False


def stop_proxy():
    """Stop the running FastAPI proxy with graceful shutdown."""
    if os.path.exists(PROXY_PID_FILE):
        try:
            with open(PROXY_PID_FILE) as f:
                pid = int(f.read().strip())

            # Try graceful shutdown first
            os.kill(pid, signal.SIGTERM)
            log(f"Sent SIGTERM to FastAPI proxy (PID {pid})")

            # Wait up to 10 seconds for graceful shutdown
            for _ in range(10):
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break  # Process has exited
            else:
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                    log(f"Force killed FastAPI proxy (PID {pid})")
                except OSError:
                    pass

            os.remove(PROXY_PID_FILE)
            log(f"Stopped FastAPI proxy (PID {pid})")

        except (OSError, ValueError) as e:
            log(f"Failed to stop proxy: {e}")
            if os.path.exists(PROXY_PID_FILE):
                os.remove(PROXY_PID_FILE)


def update_proxy(ip, port, config):
    """Start the FastAPI proxy with the given pod configuration."""
    log("Starting FastAPI proxy...")

    # Validate IP address format
    import ipaddress

    try:
        if ip not in ["localhost", "127.0.0.1", "::1"]:
            ipaddress.ip_address(ip)
    except ValueError:
        print(f"❌ Invalid IP address format: {ip}")
        sys.exit(1)

    # Validate port
    if not isinstance(port, int) or not (1 <= port <= 65535):
        print(f"❌ Invalid port number: {port}")
        sys.exit(1)

    proxy_port = config.get("proxy_port", 8000)
    if not isinstance(proxy_port, int) or not (1024 <= proxy_port <= 65535):
        print("❌ Invalid proxy_port. Must be integer between 1024-65535")
        sys.exit(1)

    cache_dir = config.get("cache_dir", "/tmp/llm_cache")
    ssl_cert = config.get("ssl_cert")
    ssl_key = config.get("ssl_key")
    use_https = config.get("use_https", False)

    # Validate SSL configuration
    if use_https:
        if not ssl_cert or not ssl_key:
            print("❌ HTTPS enabled but ssl_cert or ssl_key not provided")
            sys.exit(1)
        if not os.path.exists(ssl_cert) or not os.path.exists(ssl_key):
            print("❌ SSL certificate or key file not found")
            sys.exit(1)

    # Check dependencies
    if not os.path.exists("proxy_fastapi.py"):
        print("❌ proxy_fastapi.py not found in current directory")
        sys.exit(1)

    try:
        import fastapi
        import httpx
        import uvicorn
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Install with: pip install fastapi httpx uvicorn")
        sys.exit(1)

    # Stop existing proxy if running
    stop_proxy()

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables for the proxy
    env = os.environ.copy()
    env["RUNPOD_ENDPOINT"] = f"http://{ip}:{port}/v1/chat/completions"
    env["CACHE_DIR"] = cache_dir

    # SSL configuration
    if use_https and ssl_cert and ssl_key:
        env["SSL_CERT"] = ssl_cert
        env["SSL_KEY"] = ssl_key
        env["USE_HTTPS"] = "true"
        protocol = "https"
    else:
        protocol = "http"

    try:
        # Start the FastAPI proxy
        process = subprocess.Popen(
            [sys.executable, "proxy_fastapi.py"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with open(PROXY_PID_FILE, "w") as f:
            f.write(str(process.pid))
        os.chmod(PROXY_PID_FILE, 0o600)

        log(f"FastAPI proxy started with PID {process.pid}")
        log(f"Proxy endpoint: {protocol}://localhost:{proxy_port}")
        log(f"Backend endpoint: {env['RUNPOD_ENDPOINT']}")
        log(f"Cache directory: {cache_dir}")
        if use_https:
            log(f"SSL enabled with cert: {ssl_cert}")
        print(f"✅ FastAPI proxy started: {protocol}://localhost:{proxy_port}")

        # Give it a moment to start up, then check health
        time.sleep(2)
        if not health_check_proxy(proxy_port):
            log("⚠️ Proxy health check failed, but process appears to be running")

    except Exception as e:
        log(f"Failed to start FastAPI proxy: {e}")
        print("❌ Failed to start FastAPI proxy.")
        sys.exit(1)


def health_check_proxy(proxy_port=8000, timeout=5):
    """Check if the FastAPI proxy is responding."""
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"http://localhost:{proxy_port}/health")
            return response.status_code == 200
    except Exception:
        log("Proxy health check failed")
        return False


async def start_pod_async(config, adjusted_runtime=None):
    """Start a new pod or serverless endpoint using the service layer."""
    if dry_run:
        print("🧪 Dry-run: Simulating pod/endpoint start.")
        return

    mode = config.get("mode", "pod")
    log(f"Starting {mode} via RunPod API...")
    deps = get_dependencies()

    try:
        if mode == "pod":
            await start_pod_mode(config, deps, adjusted_runtime)
        elif mode == "serverless":
            await start_serverless_mode(config, deps, adjusted_runtime)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'pod' or 'serverless'")

    except Exception as e:
        log(f"Failed to start {mode}: {e}")
        print(f"❌ Failed to start {mode}: {e}")
        sys.exit(1)


async def start_pod_mode(config, deps, adjusted_runtime=None):
    """Start a pod in pod mode."""
    pod_service = PodManagementService(deps)

    # Prepare pod configuration from the new config structure
    pod_config = {
        "template_id": config["pod"]["template_id"],
        "gpu_type_id": config["compute"]["gpu_type_id"],
        "gpu_count": config["compute"].get("gpu_count", 1),
        "model_name": config["model"]["name"],
        "model_path": config["model"].get("path"),
        "container_disk_gb": config["pod"].get("container_disk_gb", 20),
        "volume_gb": config["pod"].get("volume_gb", 0),
        "start_jupyter": config["pod"].get("start_jupyter", False),
        "name": config["pod"].get("name", "runpod-llm"),
        "cloud_type": config["compute"].get("cloud_type", "SECURE"),
    }

    # Add any additional environment variables
    if "env" in config["pod"]:
        pod_config["env"] = config["pod"]["env"]

    pod_id = await pod_service.create_pod(pod_config)
    log(f"Pod created: {pod_id}")
    print("⏳ Waiting for pod to become ready...")

    # Wait for pod to be ready
    initial_wait = config.get("initial_wait_seconds", 10)
    await asyncio.sleep(initial_wait)

    max_attempts = config.get("max_startup_attempts", 20)
    poll_interval = config.get("poll_interval_seconds", 5)

    for attempt in range(max_attempts):
        pod = await pod_service.get_pod_status(pod_id)
        if pod["status"] == "RUNNING" and pod.get("ip"):
            port = pod["ports"][0]["privatePort"] if pod["ports"] else 8888
            update_proxy(pod["ip"], port, config)

            start = datetime.now(timezone.utc)
            runtime = adjusted_runtime or config.get("runtime_seconds", 3600)
            end = start + timedelta(seconds=runtime)

            state = {
                "mode": "pod",
                "pod_id": pod_id,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "config_snapshot": config,
            }
            save_state(state)

            print(f"✅ Pod ready: {pod_id}")
            return

        log(f"Status: {pod['status']}. Waiting... (attempt {attempt + 1}/{max_attempts})")
        await asyncio.sleep(poll_interval)

    print("❌ Pod did not become ready in time.")
    shutdown({"mode": "pod", "pod_id": pod_id})
    sys.exit(1)


async def start_serverless_mode(config, deps, adjusted_runtime=None):
    """Start a serverless endpoint in serverless mode."""
    from .serverless_service import ServerlessService

    serverless_service = ServerlessService(deps)

    # Prepare serverless configuration from the new config structure
    endpoint_config = {
        "name": config["serverless"].get("name", "vllm-llm-endpoint"),
        "template_id": config["serverless"]["template_id"],
        "gpu_type_id": config["compute"]["gpu_type_id"],
        "gpu_count": config["compute"].get("gpu_count", 1),
        "model_name": config["model"]["name"],
        "model_path": config["model"].get("path"),
        "image_name": config["serverless"].get("image_name", "runpod/vllm:latest"),
        "container_disk_gb": config["serverless"].get("container_disk_gb", 20),
        "worker_count": config["serverless"].get("worker_count", 1),
        "min_workers": config["serverless"].get("min_workers", 0),
        "max_workers": config["serverless"].get("max_workers", 3),
        "idle_timeout": config["serverless"].get("idle_timeout", 300),
        "use_model_store": config["serverless"].get("use_model_store", True),
    }

    # Add vLLM-specific configuration
    if "vllm" in config["serverless"]:
        endpoint_config.update(config["serverless"]["vllm"])

    # Add any additional environment variables
    if "env" in config["serverless"]:
        endpoint_config["env"] = config["serverless"]["env"]

    endpoint_id = await serverless_service.create_vllm_endpoint(endpoint_config)
    log(f"Serverless endpoint created: {endpoint_id}")
    print("⏳ Waiting for endpoint to become ready...")

    # Wait for endpoint to be ready
    initial_wait = config.get("initial_wait_seconds", 10)
    await asyncio.sleep(initial_wait)

    max_attempts = config.get("max_startup_attempts", 20)
    poll_interval = config.get("poll_interval_seconds", 5)

    for attempt in range(max_attempts):
        endpoint = await serverless_service.get_endpoint_status(endpoint_id)
        worker_count = endpoint.get("workerCount", 0)

        if worker_count > 0:
            # Endpoint is ready with workers
            print(f"✅ Endpoint ready: {endpoint_id} (workers: {worker_count})")

            start = datetime.now(timezone.utc)
            runtime = adjusted_runtime or config.get("runtime_seconds", 3600)
            end = start + timedelta(seconds=runtime)

            state = {
                "mode": "serverless",
                "endpoint_id": endpoint_id,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "config_snapshot": config,
            }
            save_state(state)

            print(f"✅ Serverless endpoint ready: {endpoint_id}")
            return

        log(f"Workers: {worker_count}. Waiting... (attempt {attempt + 1}/{max_attempts})")
        await asyncio.sleep(poll_interval)

    print("❌ Endpoint did not become ready in time.")
    shutdown({"mode": "serverless", "endpoint_id": endpoint_id})
    sys.exit(1)


def start_pod(config, adjusted_runtime=None):
    """Synchronous wrapper for start_pod_async."""
    asyncio.run(start_pod_async(config, adjusted_runtime))


def main():
    """Main CLI entry point."""
    # Validate environment before acquiring lock
    validate_environment()

    acquire_lock()
    try:
        config = load_config()
        state = load_state()

        if "--shutdown" in sys.argv:
            if refresh_flag:
                print("⚠️ --refresh-catalog ignored during shutdown.")
            if state:
                shutdown(state)
            else:
                print("ℹ️ No active pod to shut down.")
            return

        if state:
            mode = state.get("mode", "pod")

            # Step 1: Check expiry
            if pod_is_expired(state, config):
                print(f"⏳ {mode.title()} has expired. Shutting down.")
                shutdown(state)
                return

            # Step 2: Query status based on mode
            try:
                if mode == "pod":
                    pod_id = state.get("pod_id")
                    if pod_id:
                        # Use service layer for proper async pod status checking
                        deps = get_dependencies()
                        pod_service = PodManagementService(deps)

                        try:
                            pod_status = asyncio.run(pod_service.get_pod_status(pod_id))
                            status = pod_status.get("status")

                            if status == "RUNNING":
                                ip = pod_status.get("ip")
                                ports = pod_status.get("ports", [])
                                if ip and ports:
                                    port = ports[0].get("privatePort", 8888) if ports else 8888
                                    print(f"✅ Pod {pod_id} is healthy and running at {ip}:{port}")
                                    return
                                else:
                                    print(
                                        f"⚠️ Pod {pod_id} is running but missing IP/port information"
                                    )
                                    return
                            elif status in ["TERMINATED", "FAILED", "CANCELLED"]:
                                # Restart pod
                                end = datetime.fromisoformat(state["end_time"])
                                remaining = max(
                                    (end - datetime.now(timezone.utc)).total_seconds(), 300
                                )
                                adjusted_config = config.copy()
                                adjusted_config["runtime_seconds"] = int(remaining)
                                log(
                                    f"🔁 Pod {pod_id} was {status.lower()}. Restarting with adjusted runtime: {int(remaining)} seconds"
                                )
                                os.remove(STATE_PATH)
                                start_pod(adjusted_config)
                                return
                            else:
                                print(f"ℹ️ Pod {pod_id} status is '{status}'. No action taken.")
                                return
                        except Exception as e:
                            log(f"Error checking pod status: {e}")
                            print(f"⚠️ Could not check pod {pod_id} status: {e}")
                            print("🌐 Network issue detected. Skipping restart.")
                            return
                    else:
                        print("⚠️ No pod_id in state. Starting new pod.")
                        start_pod(config)
                elif mode == "serverless":
                    endpoint_id = state.get("endpoint_id")
                    if endpoint_id:
                        # Use service layer for proper async endpoint status checking
                        deps = get_dependencies()
                        serverless_service = ServerlessService(deps)

                        try:
                            endpoint_status = asyncio.run(
                                serverless_service.get_endpoint_status(endpoint_id)
                            )
                            worker_count = endpoint_status.get("workerCount", 0)

                            if worker_count > 0:
                                print(
                                    f"✅ Serverless endpoint {endpoint_id} is healthy with {worker_count} worker(s)"
                                )
                                return
                            else:
                                print(
                                    f"⚠️ Serverless endpoint {endpoint_id} has no active workers. Starting new endpoint."
                                )
                                start_pod(config)
                                return
                        except Exception as e:
                            log(f"Error checking endpoint status: {e}")
                            print(f"⚠️ Could not check endpoint {endpoint_id} status: {e}")
                            print("🌐 Network issue detected. Skipping restart.")
                            return
            except Exception as e:
                log(f"⚠️ Error querying {mode} status: {e}")
                print("🌐 Network issue detected. Skipping restart.")
                return

        # Starting a new pod/endpoint
        catalog = refresh_catalog() if refresh_flag else None
        if catalog:
            validate_config(config, catalog)

        start_pod(config)

    finally:
        release_lock()


if __name__ == "__main__":
    main()
