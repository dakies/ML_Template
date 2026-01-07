# ML Template - Production Training Image
# Multi-stage build for optimized image size

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Install system dependencies (minimal - uv will handle Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install Python 3.11 using uv
RUN uv python install 3.11

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (uv will use the Python it installed)
RUN uv sync --frozen --no-dev --python 3.11

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs

# Set environment for the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Default command
CMD ["python", "-m", "ml_template.train"]

# -----------------------------------------------------------
# Development stage with additional tools
FROM base AS dev

# Install dev dependencies
RUN uv sync --frozen

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy tests
COPY tests/ ./tests/

CMD ["/bin/bash"]

