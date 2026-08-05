# Karkive: container image for any web host (Render, Railway, Fly, Cloud Run,
# Hugging Face Spaces with the Docker SDK, etc.)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source, including the pre-trained model in artifacts/. No training runs
# at build or boot, so the image builds fast and the container starts in seconds.
COPY . .

# Drop root. The app only ever reads its own files — it writes nothing and, since
# training can't be triggered from a request (see model/predict.py), it never
# needs to create artifacts either. --system: no login, no home, no password.
RUN adduser --system --group --no-create-home karkive \
    && chown -R karkive:karkive /app
USER karkive

EXPOSE 8000

# Hosts inject the port via $PORT; default to 8000 for local `docker run`.
# --proxy-headers so the app sees the real client IP behind the host's TLS
# terminator; without it every request looks like it comes from the proxy and
# the rate limiter in app.py would bucket all traffic together.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
