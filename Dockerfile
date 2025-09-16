FROM ghcr.io/astral-sh/uv:debian-slim

COPY --link .python-version  pyproject.toml  uv.lock /app/

WORKDIR /app

RUN uv sync --frozen --no-cache

COPY src /app/

ENTRYPOINT ["uv", "run", "python", "-m", "windpomp"]
CMD ["clipworker"]

