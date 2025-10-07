#!/usr/bin/env python3
"""
Security utilities for RunPod LLM Manager
Provides SBOM generation, dependency scanning, and security compliance tools
"""

import os
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any


def generate_sbom(output_file: str = "sbom.json") -> bool:
    """
    Generate Software Bill of Materials using cyclonedx
    Requires: pip install cyclonedx-bom
    """
    # Input validation
    if not output_file or not isinstance(output_file, str):
        print("❌ Invalid output file path")
        return False

    # Prevent path traversal attacks
    if ".." in output_file or not output_file.endswith(".json"):
        print("❌ Invalid output file format (must be .json)")
        return False

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file) or "."
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"❌ Cannot create output directory: {e}")
        return False

    try:
        print("🔍 Generating Software Bill of Materials (SBOM)...")

        # Use cyclonedx to generate SBOM from requirements.txt
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cyclonedx_py",
                "requirements",
                "requirements.txt",
                "--output-format",
                "json",
                "--output-file",
                output_file,
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            print(f"✅ SBOM generated successfully: {output_file}")

            # Add metadata
            print("📝 Adding compliance metadata...")
            with open(output_file, "r") as f:
                sbom_data = json.load(f)

            # Update metadata to comply with CycloneDX schema
            sbom_data["metadata"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tools": [{"vendor": "CycloneDX", "name": "cyclonedx-py", "version": "latest"}],
                "component": {
                    "type": "application",
                    "name": "runpod-llm-manager",
                    "version": "1.0.0",
                    "description": "RunPod LLM Pod Manager with FastAPI Proxy - Individual Developer Project",
                    "licenses": [{"license": {"id": "MIT"}}],
                    "supplier": {
                        "name": "Individual Developer",
                        "contact": [{"name": "Individual Developer"}],
                    },
                },
                "properties": [
                    {
                        "name": "cdx:compliance:eu_cyber_resilience_act",
                        "value": "Individual developer exemption applies",
                    },
                    {"name": "cdx:security:headers", "value": "Implemented"},
                    {"name": "cdx:security:rate_limiting", "value": "Implemented"},
                    {"name": "cdx:security:input_validation", "value": "Implemented"},
                ],
            }

            with open(output_file, "w") as f:
                json.dump(sbom_data, f, indent=2)

            return True
        else:
            print(f"❌ SBOM generation failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ cyclonedx-bom not installed. Install with: pip install cyclonedx-bom")
        return False
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        return False


def scan_dependencies() -> Dict[str, Any]:
    """
    Scan dependencies for vulnerabilities using safety
    Requires: pip install safety
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "safety",
        "vulnerabilities_found": 0,
        "vulnerabilities": [],
        "status": "unknown",
    }

    try:
        print("🔍 Scanning dependencies for vulnerabilities...")

        result = subprocess.run(
            [sys.executable, "-m", "safety", "check", "--output", "json"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ No vulnerabilities found")
            results["status"] = "clean"
        else:
            # Parse safety output
            try:
                safety_output = json.loads(result.stdout)
                results["vulnerabilities"] = safety_output
                results["vulnerabilities_found"] = len(safety_output)
                results["status"] = "vulnerabilities_found"
                print(f"⚠️ Found {len(safety_output)} vulnerabilities")
            except json.JSONDecodeError:
                results["status"] = "error"
                results["error"] = result.stdout
                print(f"❌ Error parsing safety output: {result.stdout}")

    except FileNotFoundError:
        print("❌ safety not installed. Install with: pip install safety")
        results["status"] = "tool_missing"
    except Exception as e:
        print(f"❌ Error scanning dependencies: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def check_licenses() -> Dict[str, Any]:
    """
    Check dependency licenses for compliance
    Requires: pip install pip-licenses
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "pip-licenses",
        "licenses": [],
        "status": "unknown",
    }

    try:
        print("🔍 Checking dependency licenses...")

        result = subprocess.run(
            ["pip-licenses", "--format", "json"], capture_output=True, text=True
        )

        if result.returncode == 0:
            license_data = json.loads(result.stdout)
            results["licenses"] = license_data
            results["status"] = "success"

            print(f"🔍 Analyzing {len(license_data)} package licenses...")

            # Check for problematic licenses
            problematic = []
            for pkg in license_data:
                license_name = pkg.get("License", "").lower()
                if (
                    any(term in license_name for term in ["gpl", "agpl", "lgpl"])
                    and "exception" not in license_name
                ):
                    problematic.append(pkg["Name"])

            if problematic:
                print(
                    f"⚠️ Found {len(problematic)} packages with copyleft licenses: {', '.join(problematic)}"
                )
                results["copyleft_packages"] = problematic
            else:
                print("✅ No problematic licenses found")

        else:
            print(f"❌ License check failed: {result.stderr}")
            results["status"] = "error"

    except FileNotFoundError:
        print("❌ pip-licenses not installed. Install with: pip install pip-licenses")
        results["status"] = "tool_missing"
    except Exception as e:
        print(f"❌ Error checking licenses: {e}")
        results["status"] = "error"

    return results


def generate_security_report() -> Dict[str, Any]:
    """
    Generate comprehensive security report
    """
    print("🔒 Generating comprehensive security report...")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "runpod-llm-manager",
        "security_assessment": {
            "sbom_generated": generate_sbom(),
            "dependency_scan": scan_dependencies(),
            "license_check": check_licenses(),
        },
        "implemented_security_features": [
            "Security headers (XSS, CSRF, Content-Type protection)",
            "Rate limiting (60 requests/minute per IP)",
            "Input validation and sanitization",
            "HTTPS enforcement when enabled",
            "CORS protection",
            "Security event logging",
            "Request size limits",
            "Content length validation",
        ],
        "compliance_notes": [
            "EU Cyber Resilience Act: Individual developer exemption likely applies",
            "Non-commercial software: Reduced regulatory requirements",
            "GDPR: Minimal PII handling, no persistent user data storage",
            "Open source dependencies: All mainstream with active maintenance",
        ],
    }

    # Save report
    with open("security_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Security report saved to security_report.json")
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "sbom":
            generate_sbom()
        elif command == "scan":
            result = scan_dependencies()
            print(json.dumps(result, indent=2))
        elif command == "licenses":
            result = check_licenses()
            print(json.dumps(result, indent=2))
        elif command == "report":
            generate_security_report()
        else:
            print("Usage: python security_utils.py [sbom|scan|licenses|report]")
    else:
        generate_security_report()
