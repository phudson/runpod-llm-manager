# Security Documentation for RunPod LLM Manager

## Overview

This document outlines the security measures implemented in the RunPod LLM Manager to ensure compliance with EU regulations and best practices for non-commercial software developed by individual developers.

## EU Regulatory Compliance

### Cyber Resilience Act (CRA)
- **Status**: Individual developer exemption likely applies
- **Rationale**: Non-commercial software, small scale, maintained by individual
- **Implemented**: Proactive security measures for future compliance readiness

### GDPR Compliance
- **Data Handling**: Minimal PII collection (IP addresses for rate limiting only)
- **Data Retention**: No persistent user data storage
- **Data Processing**: All processing is transient and for service operation only

### Network and Information Systems (NIS2) Directive
- **Status**: Not applicable (non-critical infrastructure, individual developer)
- **Monitoring**: Basic security event logging implemented

## Implemented Security Features

### 🔒 Input Validation & Sanitization
- **Pydantic Models**: Strict input validation for all API endpoints
- **Content Sanitization**: Removal of potentially harmful patterns (script tags, etc.)
- **Size Limits**: Request size limits (1MB) and content length validation (50KB)
- **Type Validation**: Strict typing for all input parameters

### 🛡️ Rate Limiting
- **Algorithm**: Sliding window rate limiting
- **Limits**: 60 requests per minute per IP address
- **Headers**: RFC-compliant rate limit headers (`X-RateLimit-*`)
- **Response**: 429 status with retry information

### 🔐 Security Headers
- **XSS Protection**: `X-XSS-Protection: 1; mode=block`
- **Content Sniffing**: `X-Content-Type-Options: nosniff`
- **Frame Options**: `X-Frame-Options: DENY`
- **Referrer Policy**: `Referrer-Policy: strict-origin-when-cross-origin`
- **Permissions Policy**: Restrictive permissions policy

### 🔒 HTTPS Enforcement
- **HSTS**: HTTP Strict Transport Security when HTTPS is enabled
- **SSL/TLS**: Full SSL support with configurable certificates
- **Default HTTP**: HTTP with security headers (configurable)

### 🌐 CORS Protection
- **Restricted Origins**: Only allowed origins can make cross-origin requests
- **Methods**: Restricted to GET, POST
- **Credentials**: Properly configured credential handling

### 📊 Security Monitoring & Logging
- **Structured Logging**: JSON-formatted security events
- **Event Types**:
  - Rate limit violations
  - Invalid input attempts
  - Large request attempts
  - Server errors
- **Log Fields**: Client IP, User-Agent, request path, error details

### 🏗️ Architecture Security
- **No Privileged Operations**: Runs without sudo/root privileges
- **File Permissions**: Restricted permissions on sensitive files
- **Process Isolation**: Separate processes for proxy and management
- **Environment Variables**: Sensitive data via environment variables only

## Dependency Security

### Approved Dependencies
All dependencies are mainstream with active security maintenance:

| Dependency | Purpose | Security Status |
|------------|---------|-----------------|
| `fastapi` | Web framework | ✅ Active maintenance, large community |
| `httpx` | HTTP client | ✅ Modern, well-maintained |
| `uvicorn` | ASGI server | ✅ Popular, Encode-maintained |
| `pydantic` | Data validation | ✅ Core dependency, excellent security |
| `aiofiles` | Async file ops | ✅ Well-maintained |

### Dependency Management
```bash
# Regular security scanning
python security_utils.py scan

# License compliance checking
python security_utils.py licenses

# SBOM generation
python security_utils.py sbom
```

## Security Configuration

### Environment Variables
```bash
# Rate limiting
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60

# Request limits
MAX_REQUEST_SIZE=1048576  # 1MB

# HTTPS configuration
USE_HTTPS=true
SSL_CERT=/path/to/cert.pem
SSL_KEY=/path/to/key.pem
```

### CORS Configuration
```python
# In proxy_fastapi.py
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080"
]
```

## Security Testing

### Automated Security Checks
```bash
# Run comprehensive security assessment
python security_utils.py report

# Check for vulnerabilities
python security_utils.py scan

# Validate SBOM
python security_utils.py sbom
```

### Manual Security Testing
```bash
# Test rate limiting
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "test"}]}'

# Test input validation
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}'

# Check security headers
curl -I http://localhost:8000/health
```

## Incident Response

### Security Event Monitoring
- Security events are logged with structured JSON format
- Monitor logs for suspicious patterns:
  - Repeated rate limit violations
  - Invalid input attempts
  - Unusual request patterns

### Response Procedures
1. **Rate Limit Violations**: Monitor for DDoS patterns, consider IP blocking
2. **Invalid Input**: Check for attempted exploits, update validation rules
3. **Server Errors**: Investigate root cause, apply fixes
4. **Large Requests**: Monitor for resource exhaustion attempts

### Contact Information
For security issues or vulnerability reports:
- **Email**: [Your contact email]
- **Response Time**: Within 48 hours for critical issues
- **Disclosure**: Responsible disclosure encouraged

## Future Compliance Considerations

### CRA Readiness
- SBOM generation implemented
- Security update process documented
- Vulnerability reporting capability in place

### Continuous Improvement
- Regular dependency updates
- Security scanning in CI/CD pipeline
- Community monitoring for new threats

## Compliance Evidence

### SBOM (Software Bill of Materials)
Generated using CycloneDX format for transparency and compliance.

### Vulnerability Scanning
Automated scanning using Safety DB for known vulnerabilities.

### License Compliance
All dependencies checked for license compatibility.

---

*This security documentation is maintained as part of the RunPod LLM Manager project. Last updated: 2025-01-07*