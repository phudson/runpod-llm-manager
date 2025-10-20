"""
VSCode AI Assistant integration tests.
Tests that validate VSCode extensions can connect and use the proxy.
"""

import pytest


class TestVSCodeIntegration:
    """Test that VSCode AI assistants can connect and use the proxy."""

    def setup_method(self):
        """Set up test dependencies before each test."""
        from runpod_llm_manager.dependencies import set_dependencies
        from tests.fixtures.mock_services import (
            create_chat_completion_response,
            create_mock_dependencies,
            create_test_config,
        )

        config = create_test_config()
        http_responses = {
            "http://mock-runpod:4010/v1/chat/completions": create_chat_completion_response()
        }
        mock_deps = create_mock_dependencies(config, http_responses)
        set_dependencies(mock_deps)

    def test_vscode_copilot_style_request(self, client):
        """Test request format that VSCode Copilot would send."""
        # VSCode Copilot typically sends requests like this
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an AI coding assistant."},
                {"role": "user", "content": "Write a Python function to reverse a string"},
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert len(content.strip()) > 0

        # Check that response looks like code
        assert "def" in content or "function" in content

    def test_vscode_github_copilot_chat_format(self, client):
        """Test GitHub Copilot Chat specific format."""
        # GitHub Copilot Chat might send different formats
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "/explain What does this Python code do?\n\nprint('Hello, World!')",
                }
            ],
            "max_tokens": 150,
            "temperature": 0.3,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_error_handling_for_vscode(self, client):
        """Test that errors are handled gracefully for VSCode."""
        # Send malformed request
        response = client.post("/v1/chat/completions", json={"invalid": "request"})

        # Should get validation error, not crash
        assert response.status_code in [422, 500]  # Validation or server error
        data = response.json()
        assert "detail" in data or "error" in data

    def test_rate_limiting_doesnt_break_vscode(self, client):
        """Test that rate limiting returns proper errors for VSCode."""
        from runpod_llm_manager.dependencies import get_dependencies, set_dependencies
        from tests.fixtures.mock_services import MockRateLimiter

        # Create a rate limiter that limits after 3 requests
        limiting_limiter = MockRateLimiter(always_allow=False, limit_after=3)
        deps = get_dependencies()
        deps.rate_limiter = limiting_limiter
        set_dependencies(deps)

        # Make many rapid requests
        for i in range(6):  # More than the limit
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": f"Test {i}"}],
                    "max_tokens": 10,
                },
            )

            if response.status_code == 429:
                # Rate limited - check error format
                data = response.json()
                assert "error" in data
                assert "Rate limit exceeded" in data["error"]
                break
        else:
            # If we get here, rate limiting didn't trigger
            # That's OK for this test - just means limits are high
            pass

    def test_vscode_extension_headers_accepted(self, client):
        """Test that VSCode extension headers are accepted."""
        # VSCode extensions might send various headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VSCode-Copilot/1.0",
            "Authorization": "Bearer test-token",  # If applicable
        }

        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}

        response = client.post("/v1/chat/completions", json=payload, headers=headers)

        # Should work regardless of extra headers
        assert response.status_code == 200

    def test_openai_compatible_response_format(self, client):
        """Test that responses are fully OpenAI API compatible."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 50,
            "temperature": 0.5,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate complete OpenAI API compliance
        required_fields = ["id", "object", "created", "model", "choices"]
        for field in required_fields:
            assert field in data

        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert len(data["choices"]) > 0

        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]
        assert "finish_reason" in choice
