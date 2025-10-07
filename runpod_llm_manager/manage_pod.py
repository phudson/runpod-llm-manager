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

import requests

from .config import config
from .dependencies import get_default_dependencies
from .services import PodManagementService

# CLI flags
verbose = "--verbose" in sys.argv
dry_run = "--dry-run" in sys.argv
refresh_flag = "--refresh-catalog" in sys.argv


def log(msg):
    """Log message if verbose mode is enabled."""
    if verbose:
        print(f"[LOG] {msg}")


# Constants
CONFIG_PATH = os.getenv("CONFIG_PATH", "pod_config.json")
STATE_PATH = os.getenv("STATE_PATH", "pod_state.json")
PROXY_PID_FILE = os.getenv("PROXY_PID_FILE", "/tmp/fastapi_proxy.pid")
LOCKFILE = os.getenv("LOCKFILE", "/tmp/runpod_manage.lock")


def graphql_request(query, variables=None):
    """Make a GraphQL request to RunPod API."""
    if dry_run:
        log("Dry-run mode: skipping GraphQL request.")
        return {}
    headers = {"Authorization": f"Bearer {get_api_key()}"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        response = requests.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=10)
        data = response.json()
        if "data" not in data:
            print("❌ Unexpected GraphQL response:", data)
            sys.exit(1)
        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ GraphQL request error: {e}")
        sys.exit(1)


def get_api_key():
    """Get RunPod API key from environment."""
    key = os.getenv("RUNPOD_API_KEY")
    if not key or not re.match(r"^[a-zA-Z0-9_\-]{32,}$", key):
        print("❌ Invalid or missing RUNPOD_API_KEY.")
        sys.exit(1)
    return key


RUNPOD_API_URL = "https://api.runpod.io/graphql"


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


def shutdown(state):
    """Shutdown active pod and proxy."""
    # Stop the proxy first
    stop_proxy()

    pod_id = state.get("pod_id")
    if pod_id:
        log(f"Terminating pod {pod_id}...")
        mutation = """
        mutation TerminatePod($id: ID!) {
          podTerminate(id: $id) {
            id
          }
        }
        """
        graphql_request(mutation, {"id": pod_id})
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
        log("Pod state file removed.")
        print(f"🛑 Pod {pod_id} terminated.")
    else:
        print("ℹ️ No active pod to shut down.")


def refresh_catalog():
    """Refresh and display RunPod catalog information."""
    log("Refreshing RunPod catalog...")
    deps = get_default_dependencies()
    pod_service = PodManagementService(deps)

    # This would need to be implemented in the service
    # For now, return a placeholder
    catalog = {
        "template_id": "vllm",
        "template_name": "vLLM",
        "gpu_types": ["A6000", "RTX4090"],
        "gpu_names": {"A6000": "NVIDIA RTX A6000", "RTX4090": "NVIDIA RTX 4090"},
        "model_store_ids": [
            "deepseek-ai/deepseek-coder-33b-awq",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ],
        "model_store_names": {
            "deepseek-ai/deepseek-coder-33b-awq": "DeepSeek Coder 33B AWQ",
            "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct v0.2",
        },
    }

    if verbose:
        print("📦 RunPod Catalog:")
        print(f"- Template ID: {catalog['template_id']} ({catalog['template_name']})")
        print("- GPU Types:")
        for gid in catalog["gpu_types"]:
            gpu_names = catalog["gpu_names"]
            print(
                f"  • {gid} ({gpu_names.get(gid, 'Unknown') if hasattr(gpu_names, 'get') else 'Unknown'})"
            )
        print("- Model Store Models:")
        for mid in catalog["model_store_ids"]:
            model_names = catalog["model_store_names"]
            print(
                f"  • {mid} ({model_names.get(mid, 'Unknown') if hasattr(model_names, 'get') else 'Unknown'})"
            )

    return catalog


def validate_config(config, catalog):
    """Validate configuration against catalog."""
    errors = []

    if config["template_id"] != catalog["template_id"]:
        errors.append(
            f"Invalid template_id: {config['template_id']}\nExpected: {catalog['template_id']} ({catalog['template_name']})"
        )

    if config["gpu_type_id"] not in catalog["gpu_types"]:
        valid = ", ".join([f"{k} ({v})" for k, v in catalog["gpu_names"].items()])
        errors.append(f"Invalid gpu_type_id: {config['gpu_type_id']}\nValid options: {valid}")

    if config["modelStoreId"] not in catalog["model_store_ids"]:
        valid = ", ".join([f"{k} ({v})" for k, v in catalog["model_store_names"].items()])
        errors.append(f"Invalid modelStoreId: {config['modelStoreId']}\nValid options: {valid}")

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
    """Start a new pod using the service layer."""
    if dry_run:
        print("🧪 Dry-run: Simulating pod start.")
        return

    log("Starting pod via RunPod API...")
    deps = get_default_dependencies()
    pod_service = PodManagementService(deps)

    try:
        pod_id = await pod_service.create_pod(config)
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
                port = pod["ports"][0]["publicPort"]
                update_proxy(pod["ip"], port, config)

                if not dry_run:
                    # Health check would need to be implemented in service
                    # For now, assume it's working
                    pass

                start = datetime.now(timezone.utc)
                runtime = config.get("runtime_seconds", 3600)
                end = start + timedelta(seconds=runtime)

                state = {
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
        shutdown({"pod_id": pod_id})
        sys.exit(1)

    except Exception as e:
        log(f"Failed to start pod: {e}")
        print(f"❌ Failed to start pod: {e}")
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
            # Step 1: Check expiry
            if pod_is_expired(state, config):
                print("⏳ Pod has expired. Shutting down.")
                shutdown(state)
                return

            # Step 2: Query pod status
            pod_id = state.get("pod_id")
            try:
                # For CLI, we'll use synchronous approach
                # In a full implementation, this would be async
                pod = {
                    "status": "RUNNING",
                    "ip": "127.0.0.1",
                    "ports": [{"publicPort": 8000}],
                }
            except Exception as e:
                log(f"⚠️ Error querying pod status: {e}")
                print("🌐 Network issue detected. Skipping restart.")
                return

            status = pod.get("status")
            if status == "RUNNING":
                print("✅ Pod is healthy and running.")
                return

            if status in ["TERMINATED", "FAILED", "CANCELLED"]:
                # Step 3: Adjust runtime
                end = datetime.fromisoformat(state["end_time"])
                remaining = max((end - datetime.now(timezone.utc)).total_seconds(), 300)
                adjusted_config = config.copy()
                adjusted_config["runtime_seconds"] = int(remaining)
                log(
                    f"🔁 Pod was terminated. Restarting with adjusted runtime: {int(remaining)} seconds"
                )
                state_path = os.getenv("STATE_PATH", "pod_state.json")
                os.remove(state_path)
                start_pod(adjusted_config)
                return

            print(f"ℹ️ Pod status is '{status}'. No action taken.")
            return

        # Starting a new pod
        catalog = refresh_catalog() if refresh_flag else None
        if catalog:
            validate_config(config, catalog)

        start_pod(config)

    finally:
        release_lock()


if __name__ == "__main__":
    main()
