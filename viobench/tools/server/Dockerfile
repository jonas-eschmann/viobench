# The submission server's image, for both services in compose.yaml next to this file. Build context: the repository root.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 VIOBENCH_SERVER_DIR=/srv/server-data
WORKDIR /opt/viobench
COPY pyproject.toml README.md ./
COPY viobench viobench
# server-data is created here so that a fresh named volume mounted on it is owned by the unprivileged user
RUN pip install --no-cache-dir ".[server]" \
 && useradd --system --create-home --uid 10001 viobench \
 && mkdir -p /srv/server-data && chown viobench /srv/server-data
USER viobench
