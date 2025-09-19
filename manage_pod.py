#!/usr/bin/env python3

import os
import json
import sys
import time
import requests
import re
import signal
import subprocess
from datetime import datetime, timedelta, timezone

CONFIG_PATH = "pod_config.json"
STATE_PATH = "pod_state.json"
LOCKFILE = "/tmp/runpod_manage.lock"
PROXY_PID_FILE = "/tmp/fastapi_proxy.pid"
RUNPOD_API_URL = "https://api.runpod.io/graphql"

verbose = "--verbose" in sys.argv
dry_run = "--dry-run" in sys.argv
refresh_flag = "--refresh-catalog" in sys.argv

def log(msg):
    if verbose:
        print(f"[LOG] {msg}")

def get_api_key():
    key = os.getenv("RUNPOD_API_KEY")
    if not key or not re.match(r"^[a-zA-Z0-9_\-]{32,}$", key):
        print("❌ Invalid or missing RUNPOD_API_KEY.")
        sys.exit(1)
    return key

def graphql_request(query, variables=None):
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

def load_config():
    log("Loading pod_config.json...")
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_state(state):
    log("Saving pod_state.json...")
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def load_state():
    if not os.path.exists(STATE_PATH):
        log("No existing pod_state.json found.")
        return None
    log("Loading pod_state.json...")
    with open(STATE_PATH) as f:
        return json.load(f)

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

def check_proxy_dependencies():
    """Verify that proxy_fastapi.py and its dependencies are available."""
    if not os.path.exists("proxy_fastapi.py"):
        print("❌ proxy_fastapi.py not found in current directory")
        sys.exit(1)

    try:
        import fastapi, httpx, uvicorn
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Install with: pip install fastapi httpx uvicorn")
        sys.exit(1)

def health_check_proxy(proxy_port=8000, timeout=5):
    """Check if the FastAPI proxy is responding."""
    try:
        response = requests.get(f"http://localhost:{proxy_port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def update_proxy(ip, port, config):
    log("Starting FastAPI proxy...")

    # Validate configuration
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
    check_proxy_dependencies()

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
        # Start the FastAPI proxy (avoid pipe blocking)
        process = subprocess.Popen(
            [sys.executable, "proxy_fastapi.py"],
            env=env,
            stdout=subprocess.DEVNULL,  # Discard output to avoid blocking
            stderr=subprocess.DEVNULL
        )

        # Save the PID with restricted permissions
        with open(PROXY_PID_FILE, "w") as f:
            f.write(str(process.pid))
        os.chmod(PROXY_PID_FILE, 0o600)  # Owner read/write only

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


def pod_is_expired(state, config):
    start = datetime.fromisoformat(state["start_time"])
    end = datetime.fromisoformat(state["end_time"])
    expired = datetime.now(timezone.utc) > end
    log(f"Pod started at {start}, expires at {end}. Expired: {expired}")
    return expired


def shutdown(state):
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
        os.remove(STATE_PATH)
        log("pod_state.json removed.")
        print(f"🛑 Pod {pod_id} terminated.")
    else:
        print("ℹ️ No active pod to shut down.")

def refresh_catalog():
    log("Refreshing RunPod catalog...")
    def query(q):
        return graphql_request(q)

    templates_q = """
    query {
      templates {
        id
        name
        imageName
      }
    }
    """
    templates = query(templates_q)["data"]["templates"]
    vllm_templates = [t for t in templates if "vllm" in t["imageName"].lower()]
    latest_template = vllm_templates[0] if vllm_templates else None

    gpus_q = """
    query {
      gpuTypes {
        id
        displayName
      }
    }
    """
    gpus = query(gpus_q)["data"]["gpuTypes"]

    models = [
        "deepseek-ai/deepseek-coder-33b-awq",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "openchat/openchat-3.5-0106",
        "Qwen/Qwen1.5-7B-Chat"
    ]

    catalog = {
        "template_id": latest_template["id"] if latest_template else None,
        "template_name": latest_template["name"] if latest_template else None,
        "gpu_types": [g["id"] for g in gpus],
        "gpu_names": {g["id"]: g["displayName"] for g in gpus},
        "models": models
    }

    if verbose:
        print("📦 RunPod Catalog:")
        print(f"- Template ID: {catalog['template_id']} ({catalog['template_name']})")
        print("- GPU Types:")
        for gid in catalog["gpu_types"]:
            print(f"  • {gid} ({catalog['gpu_names'][gid]})")
        print("- Supported Models:")
        for m in catalog["models"]:
            print(f"  • {m}")

    return catalog

def validate_config(config, catalog):
    errors = []

    if config["template_id"] != catalog["template_id"]:
        errors.append(f"Invalid template_id: {config['template_id']}\nExpected: {catalog['template_id']} ({catalog['template_name']})")

    if config["gpu_type_id"] not in catalog["gpu_types"]:
        valid = ", ".join([f"{k} ({v})" for k, v in catalog["gpu_names"].items()])
        errors.append(f"Invalid gpu_type_id: {config['gpu_type_id']}\nValid options: {valid}")

    if config["model"] not in catalog["models"]:
        valid = ", ".join(catalog["models"])
        errors.append(f"Invalid model: {config['model']}\nValid options: {valid}")

    if errors:
        print("❌ Configuration validation failed:")
        for e in errors:
            print("-", e)
        sys.exit(1)
    else:
        log("✅ Configuration validated successfully.")

def start_pod(config, adjusted_runtime=None):
    if dry_run:
        print("🧪 Dry-run: Simulating pod start.")
        return

    log("Starting pod via RunPod API...")
    mutation = """
    mutation StartPod($input: PodInput!) {
      podCreate(input: $input) {
        id
      }
    }
    """
    input_data = {
        "templateId": config["template_id"],
        "cloudType": "SECURE",
        "gpuTypeId": config["gpu_type_id"],
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "startJupyter": False,
        "name": "runpod-llm",
        "env": {
            "MODEL_NAME": config["model"]
        }
    }
    result = graphql_request(mutation, {"input": input_data})
    pod_id = result["data"]["podCreate"]["id"]
    log(f"Pod created: {pod_id}")
    print("⏳ Waiting for pod to become ready...")

    # Optimized startup: reduce initial wait and use faster polling
    initial_wait = config.get("initial_wait_seconds", 10)  # Reduced from 20
    time.sleep(initial_wait)

    # Optimized polling with configurable intervals
    max_attempts = config.get("max_startup_attempts", 20)  # Reduced from 30
    poll_interval = config.get("poll_interval_seconds", 5)  # Reduced from 10

    for attempt in range(max_attempts):
        pod = get_pod_status(pod_id)
        if pod["status"] == "RUNNING" and pod["ip"]:
            port = pod["ports"][0]["publicPort"]
            update_proxy(pod["ip"], port, config)
            if not dry_run and not health_check(pod["ip"], port, config["model"]):
                print("❌ Pod endpoint failed health check.")
                shutdown({"pod_id": pod_id})
                sys.exit(1)

            start = datetime.now(timezone.utc)
            runtime = config.get("runtime_seconds", 3600)
            end = start + timedelta(seconds=runtime)

            state = {
                "pod_id": pod_id,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "config_snapshot": config
            }
            save_state(state)

            print(f"✅ Pod ready: {pod_id}")
            return
        log(f"Status: {pod['status']}. Waiting... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(poll_interval)

    print("❌ Pod did not become ready in time.")
    shutdown({"pod_id": pod_id})
    sys.exit(1)


def get_pod_status(pod_id):
    query = """
    query PodStatus($id: ID!) {
      pod(id: $id) {
        id
        status
        ip
        ports {
          ip
          privatePort
          publicPort
        }
      }
    }
    """
    result = graphql_request(query, {"id": pod_id})
    return result["data"]["pod"]

def health_check(ip, port, model, retries=2):  # Reduced from 3
    """Optimized health check with faster timeout and reduced retries."""
    url = f"http://{ip}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],  # Shorter message
        "max_tokens": 10,  # Limit response size
        "temperature": 0.1  # Lower temperature for faster response
    }

    for attempt in range(retries):
        try:
            # Faster timeout for quicker failure detection
            response = requests.post(url, json=payload, timeout=3)  # Reduced from 5
            if response.status_code == 200:
                log(f"✅ Health check passed on attempt {attempt + 1}.")
                return True
            else:
                log(f"⚠️ Health check failed with status {response.status_code}.")
        except Exception as e:
            log(f"⚠️ Health check exception on attempt {attempt + 1}: {e}")
        time.sleep(1)  # Reduced from 2
    return False

def acquire_lock():
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
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)
        log("🔓 Lockfile released.")

def validate_environment():
    """Validate required environment variables."""
    if not os.getenv("RUNPOD_API_KEY"):
        print("❌ RUNPOD_API_KEY environment variable not set")
        print("Set it with: export RUNPOD_API_KEY='your-api-key'")
        sys.exit(1)

def main():
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
                pod = get_pod_status(pod_id)
            except Exception as e:
                log(f"⚠️ Failed to query pod status: {e}")
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
                log(f"🔁 Pod was terminated. Restarting with adjusted runtime: {int(remaining)} seconds")
                os.remove(STATE_PATH)
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
