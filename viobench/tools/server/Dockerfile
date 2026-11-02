# The submission server's image, for both services in compose.yaml next to this file. Build context: the repository root.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 VIOBENCH_SERVER_DIR=/srv/server-data CONTA_CACHE=/srv/conta
WORKDIR /opt/viobench
COPY pyproject.toml README.md ./
COPY viobench viobench
# server-data and conta (the episode cache) are created here so that fresh named volumes mounted on them are owned by
# the unprivileged user
RUN pip install --no-cache-dir ".[server]" \
 && useradd --system --create-home --uid 10001 viobench \
 && mkdir -p /srv/server-data /srv/conta && chown viobench /srv/server-data /srv/conta
USER viobench
