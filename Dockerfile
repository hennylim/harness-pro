# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r harness && useradd -r -g harness harness

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

# Create workspace and runs directories with correct ownership
RUN mkdir -p sandbox_workspace runs \
    && chown -R harness:harness /app

USER harness

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json

ENTRYPOINT ["python", "main.py"]
