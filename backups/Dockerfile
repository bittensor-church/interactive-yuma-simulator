FROM postgres:17
WORKDIR /root
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/
COPY --from=getsentry/sentry-cli:3.8.2 /bin/sentry-cli /bin/
# Python comes from uv below: the base image distro no longer ships python3.11
RUN apt-get update && apt-get install -y cron zstd
COPY --chmod=0755 docker-entrypoint.sh /
COPY --chmod=0644 cron.d/backup /etc/cron.d/backup
COPY bin ./
RUN uv venv --python 3.11 .venv && uv pip install -r requirements.txt

CMD ["/docker-entrypoint.sh"]