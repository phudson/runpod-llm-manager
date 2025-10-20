#!/bin/bash
# test-functional.sh - Run functional tests locally

set -e  # Exit on any error

echo "🧪 Running Functional Tests Locally"
echo "==================================="

# Function to run a test suite
run_test_suite() {
    local test_name="$1"
    local test_file="$2"

    echo ""
    echo "🔍 Running $test_name tests..."
    echo "----------------------------------------"

    if pytest "tests/functional/$test_file" -v --tb=short; then
        echo "✅ $test_name tests passed"
        return 0
    else
        echo "❌ $test_name tests failed"
        return 1
    fi
}

# Run core functionality tests
run_test_suite "Core Functionality" "test_core_functionality.py"

# Run watchdog/monitoring tests
run_test_suite "Watchdogs & Monitoring" "test_watchdogs.py"

# Run VSCode integration tests
run_test_suite "VSCode Integration" "test_vscode_integration.py"

# Run RunPod integration tests (only if API key is available)
if [ -n "$RUNPOD_API_KEY" ]; then
    echo ""
    echo "🔑 RunPod API key detected - running integration tests..."
    run_test_suite "RunPod Integration" "test_runpod_integration.py"
else
    echo ""
    echo "⏭️  Skipping RunPod integration tests (no API key)"
    echo "   Set RUNPOD_API_KEY environment variable to run these tests"
fi

echo ""
echo "🎉 Functional testing completed!"
echo ""
echo "To run individual test suites:"
echo "  pytest tests/functional/test_core_functionality.py -v"
echo "  pytest tests/functional/test_watchdogs.py -v"
echo "  pytest tests/functional/test_vscode_integration.py -v"
echo "  RUNPOD_API_KEY=your_key pytest tests/functional/test_runpod_integration.py -v"
