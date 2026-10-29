# The submission server's image, for both services in compose.yaml next to this file. Build context: the repository root.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 RELPOSE_SERVER_DIR=/srv/server-data
WORKDIR /opt/relpose
COPY pyproject.toml README.md ./
COPY relpose relpose
# server-data is created here so that a fresh named volume mounted on it is owned by the unprivileged user
RUN pip install --no-cache-dir ".[server,data]" \
 && useradd --system --create-home --uid 10001 relpose \
 && mkdir -p /srv/server-data && chown relpose /srv/server-data
USER relpose
