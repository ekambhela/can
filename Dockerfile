# Karkive — container image for any web host (Render, Railway, Fly, Cloud Run,
# Hugging Face Spaces with the Docker SDK, etc.)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source.
COPY . .

# Train the model at build time so the running container boots fast
# (otherwise the model self-trains on the first request, ~30s).
RUN python -m model.train

EXPOSE 8000

# Hosts inject the port via $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
