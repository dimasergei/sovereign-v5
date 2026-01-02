# =============================================================================
# INSTITUTIONAL PROP TRADING SYSTEM - DOCKER IMAGE
# =============================================================================
# Multi-stage build for optimized production image

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Sovereign Trading System"
LABEL version="5.0"
LABEL description="Institutional-grade prop trading system"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME="/app" \
    CONFIG_DIR="/app/config" \
    DATA_DIR="/app/data" \
    LOGS_DIR="/app/logs" \
    MODELS_DIR="/app/models_saved"

# Create non-root user for security
RUN groupadd -r trader && useradd -r -g trader trader

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directories
WORKDIR ${APP_HOME}
RUN mkdir -p ${CONFIG_DIR} ${DATA_DIR} ${LOGS_DIR} ${MODELS_DIR}

# Copy application code
COPY --chown=trader:trader . .

# Set permissions
RUN chown -R trader:trader ${APP_HOME}

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Expose ports
# 9090 - Prometheus metrics
# 8080 - API (if enabled)
EXPOSE 9090 8080

# Default command
CMD ["python", "-m", "main"]

# -----------------------------------------------------------------------------
# Stage 3: Development
# -----------------------------------------------------------------------------
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    ipython

USER trader

CMD ["python", "-m", "pytest", "-v"]
