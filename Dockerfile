# Two-stage build: the frontend is compiled to static assets, then served by the
# same FastAPI process that serves the API. One container, one port, no reverse
# proxy needed inside the image — Caddy sits in front of it in deployment only,
# for TLS.
#
# The resulting image makes no outbound network calls at runtime once model
# weights are present. See README.md, "Running air-gapped".

# ---------------------------------------------------------------------------
# Stage 1 — frontend
# ---------------------------------------------------------------------------
FROM node:22-slim AS web

WORKDIR /build
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# libgl1 and libglib2.0-0 are required by opencv even in its headless build.
# Installed as a discrete layer so dependency changes do not invalidate it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ocr.txt requirements-ocr-nodeps.txt constraints.txt ./
# Two steps on purpose. rapidocr-onnxruntime declares a dependency on the full
# `opencv-python` build; installing it with --no-deps, after its real
# dependencies, keeps OpenCV to the headless build that THIRD-PARTY-NOTICES.md
# commits to. See requirements-ocr.txt.
# constraints.txt locks every transitive package to the tested version too.
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt -r requirements-ocr.txt \
    && pip install --no-cache-dir --no-deps -c constraints.txt -r requirements-ocr-nodeps.txt \
    && ! pip show opencv-python >/dev/null 2>&1

COPY api/ ./api/
COPY extraction/ ./extraction/
COPY rules/ ./rules/
COPY fixtures/ ./fixtures/
COPY --from=web /build/dist ./web/dist

# Prove the OCR engine loads inside this image, and fail the build if it does
# not. The model weights ship inside the rapidocr wheel, so there is nothing to
# download — this is a check, not a warm-up.
#
# The first version of this step wrapped the import in try/except, spread the
# Python across escaped newlines, and ended with `|| true`. The escaped `\n`
# sequences reached Python as literal backslash-n, a syntax error, which
# `|| true` then swallowed: the step could never fail, so it checked nothing.
# A broken engine would have shipped as a container that boots, reports
# "degraded", and rejects every upload. It must be allowed to fail.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print('OCR engine loads')"

# Non-root. Nothing is written to disk at runtime — uploaded images are held in
# memory and discarded — so the filesystem can stay read-only in deployment.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LABEL_VERIFY_WEB_DIST=/app/web/dist

# The listening port comes from $PORT when the platform sets one — Render does,
# and routes traffic to whatever it names — and falls back to 8080 for local
# runs and docker compose. Shell form with `exec` so uvicorn replaces the shell
# as PID 1 and receives the platform's shutdown signal directly, rather than a
# shell swallowing it and the container being killed mid-request.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"$PORT\""]
