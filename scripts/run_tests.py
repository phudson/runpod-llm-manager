#!/usr/bin/env python3
import subprocess
import sys


def run_unit_tests():
    return subprocess.run([sys.executable, "-m", "pytest", "tests/unit/", "-v"])


def run_integration_tests():
    # Run with mock by default, real if env vars set
    return subprocess.run([sys.executable, "-m", "pytest", "tests/integration/", "-v"])


def run_functional_tests():
    print("⚠️ Functional tests require real credentials and will incur costs!")
    confirm = input("Continue? (y/N): ")
    if confirm.lower() == "y":
        return subprocess.run(
            [sys.executable, "-m", "pytest", "tests/functional/", "-v", "-m", "functional"]
        )
    return None
