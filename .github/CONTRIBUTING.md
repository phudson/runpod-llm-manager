# Contributing to RunPod LLM Pod Manager

Thank you for your interest in contributing to the RunPod LLM Pod Manager! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to abide by its terms. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Provide detailed information including:
  - Steps to reproduce the issue
  - Expected vs. actual behavior
  - Environment details (OS, Python version, etc.)
  - Error messages and logs

### Contributing Code

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards
4. **Run tests** and ensure they pass
5. **Update documentation** if needed
6. **Commit your changes** with clear commit messages
7. **Push to your fork** and submit a pull request

### Pull Request Process

1. Ensure your PR includes:
   - Clear description of changes
   - Reference to any related issues
   - Tests for new functionality
   - Updated documentation

2. PRs will be reviewed by maintainers who may request changes

3. Once approved, your PR will be merged

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- RunPod API key (for testing)

### Installation
```bash
git clone https://github.com/your-username/runpod-llm-manager.git
cd runpod-llm-manager
pip install -r requirements.txt
```

### Running Tests
```bash
# Run security assessment
python security_utils.py report

# Run basic functionality tests
python manage_pod.py --dry-run
```

## Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep line length under 88 characters (Black formatter default)

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable

### Security Considerations
- Never commit API keys, passwords, or sensitive data
- Use environment variables for configuration
- Follow secure coding practices outlined in [SECURITY.md](SECURITY.md)

## Testing

### Security Testing
```bash
# Run comprehensive security assessment
python security_utils.py report

# Check for vulnerabilities
python security_utils.py scan

# Validate SBOM
python security_utils.py sbom
```

### Functional Testing
- Test pod creation and management
- Verify proxy functionality
- Test VSCode extension integration
- Validate error handling

## Documentation

### Code Documentation
- Update README.md for any new features
- Update SECURITY.md for security-related changes
- Add docstrings to new functions
- Update configuration examples

### User Documentation
- Keep installation instructions current
- Update troubleshooting guides
- Maintain VSCode extension setup guides

## License Compliance

When contributing code:
- Ensure compatibility with MIT license
- Be aware of LGPL dependencies (chardet, frozendict)
- Follow license attribution requirements
- See [LICENSE](LICENSE) and license compliance section in README.md

## Getting Help

- Check existing issues and documentation first
- Use GitHub Discussions for questions
- Contact maintainers for sensitive security issues

## Recognition

Contributors will be recognized in the project documentation and GitHub's contributor insights. Thank you for helping improve the RunPod LLM Pod Manager!