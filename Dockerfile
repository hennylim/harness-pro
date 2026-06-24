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

# ── 언어별 lint 도구 설치 ──────────────────────────────────────────────────────
# Python  : ruff / flake8  (pip, Stage 1에서 설치됨)
# Bash    : shellcheck
# C/C++   : gcc, g++, clang-tidy
RUN apt-get update && apt-get install -y --no-install-recommends \
        shellcheck \
        gcc \
        g++ \
        clang-tidy \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-root
RUN groupadd -r harness && useradd -r -g harness harness

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p sandbox_workspace runs \
    && chown -R harness:harness /app

USER harness

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json \
    TARGET_LANGUAGE=python

ENTRYPOINT ["python", "main.py"]
