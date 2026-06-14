# Slim image for the Org / User service (port 8003).
# Lightweight deps only (FastAPI + jose + passlib + requests) — no LiveKit/Pinecone/OpenAI.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install pinned deps first for layer caching + reproducible builds.
# requirements.lock is the fully-resolved freeze; requirements.txt is the human-readable source.
COPY requirements.txt requirements.lock /app/user_service/
RUN pip install --no-cache-dir -r /app/user_service/requirements.lock

# Copy the package. Run context is user_service/, so contents land in /app/user_service
# and the app is importable as `user_service.main:app` (relative imports require this layout).
COPY . /app/user_service

# Non-root user.
RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

CMD ["uvicorn", "user_service.main:app", "--host", "0.0.0.0", "--port", "8003"]
