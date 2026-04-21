# syntax=docker/dockerfile:1.7
# DIX VISION v42.2 cloud image -- 24/7 cockpit + learning/sourcing.
#
#   docker build -t dix-vision:42.2 .
#   docker run --rm -p 8765:8765 -v dix-data:/data dix-vision:42.2
#
# Env:
#   DIX_MODE=cloud | worker        (default: cloud)
#   DIX_BIND_HOST=0.0.0.0          (set via launcher when mode=cloud)
#   DIX_PORT=8765
#   DIX_PUBLIC_URL=https://...     (for phone pairing QR)
#   DIX_COCKPIT_TOKEN=...          (optional; auto-generated on first boot)
#
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DIX_MODE=cloud \
    DIX_PORT=8765 \
    DIX_PAIRING_DB=/data/pairing.sqlite \
    DIX_COCKPIT_TOKEN_FILE=/data/cockpit_token.txt

RUN useradd --create-home --uid 1001 --shell /bin/bash dix \
 && mkdir -p /app /data \
 && chown -R dix:dix /app /data

WORKDIR /app

COPY --chown=dix:dix requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=dix:dix . .

USER dix
VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; \
urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2); sys.exit(0)" \
  || exit 1

ENTRYPOINT ["python", "-m", "cockpit"]
