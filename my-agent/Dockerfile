# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
# UV is a fast Python package manager that provides better performance than pip
# We use the slim variant to keep the image size smaller while still having essential tools
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Compile Python source to bytecode (.pyc) during install so the first import
# doesn't pay the compilation cost. This reduces agent cold-start time at the
# expense of a slightly longer build.
ENV UV_COMPILE_BYTECODE=1

# Cache HuggingFace models (Silero VAD + turn-detector) under /app so the files
# downloaded by `download-files` in the build stage are included when /app is
# copied into the final image (otherwise they land in /root/.cache and are lost).
ENV HF_HOME=/app/.cache/huggingface

# --- Build stage ---
# Install dependencies, build native extensions, and prepare the application
FROM base AS build

# Install build dependencies required for Python packages with native extensions
# gcc: C compiler needed for building Python packages with C extensions
# g++: C++ compiler needed for building Python packages with C++ extensions
# python3-dev: Python development headers needed for compilation
# We clean up the apt cache after installation to keep the image size down
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    nodejs \
    npm \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
# And set it as the working directory
WORKDIR /app

# Copy just the dependency files first, for more efficient layer caching
COPY pyproject.toml uv.lock ./
RUN mkdir -p src

# Install Python dependencies using UV's lock file
# --locked ensures we use exact versions from uv.lock for reproducible builds
# This creates a virtual environment and installs all dependencies
# Ensure your uv.lock file is checked in for consistency across environments
#
# The BuildKit cache mount persists uv's package cache ACROSS builds, so even if
# this layer re-runs (e.g. uv.lock changed) wheels come from the local cache
# instead of being re-downloaded from PyPI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Pre-download ML models (Silero VAD + turn detector) BEFORE copying the app
# source. This runs a STANDALONE script (src/download_models.py) that imports only
# the model plugins — not agent.py — so this layer depends ONLY on that script +
# the installed deps. A change to any OTHER application source file therefore does
# NOT invalidate this layer: the models are cached exactly like `uv sync` and are
# never re-downloaded on an app-code rebuild.
#
# The BuildKit cache mount additionally avoids a network re-download on the rare
# dep/script change that does re-run this layer.
COPY src/download_models.py ./src/download_models.py
RUN --mount=type=cache,target=/root/.cache/agent-model-dl \
    HF_HOME=/root/.cache/agent-model-dl uv run --no-sync src/download_models.py && \
    mkdir -p /app/.cache/huggingface && \
    cp -a /root/.cache/agent-model-dl/. /app/.cache/huggingface/

# Copy all remaining application files into the container. This layer is
# invalidated on any source change, but the deps + model layers above stay cached.
# (Excludes files specified in .dockerignore)
COPY . .

# --- Production stage ---
# Build tools (gcc, g++, python3-dev) are not included in the final image
FROM base

# Install Node.js/npm so `npx` is available for the GitHub MCP server subprocess.
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/build/building/best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Copy the application and virtual environment with correct ownership in a single layer
# This avoids expensive recursive chown and excludes build tools from the final image
COPY --from=build --chown=appuser:appuser /app /app

WORKDIR /app

# Switch to the non-privileged user for all subsequent operations
# This improves security by not running as root
USER appuser

# Run the AgentServer using UV
# UV will activate the virtual environment and run the agent.
# The "start" command tells the AgentServer to connect to LiveKit and begin waiting for jobs.
CMD ["uv", "run", "src/agent.py", "start"]
