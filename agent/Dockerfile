# ---- dev: VS Code Dev Containers 用 ----
FROM mcr.microsoft.com/devcontainers/base:noble AS dev

USER vscode
WORKDIR /workspace
COPY --chown=ubuntu:ubuntu build_container/ /workspace/build_container/
RUN ["/bin/bash", "build_container/init_env.sh"]

# OpenAI Compatible API endpoint port
EXPOSE 8000/tcp

# ---- prod: 本番用（uv + アプリのみ） ----
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS prod

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 依存のみ先にインストールしてレイヤーキャッシュを効かせる
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --no-install-project

COPY src/ /app/src/
COPY README.md /app/README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

RUN groupadd --system --gid 999 app \
    && useradd --system --gid app --uid 999 --create-home app \
    && chown -R app:app /app
USER app

# OpenAI Compatible API endpoint port
EXPOSE 8000/tcp
CMD ["uv", "run", "--no-dev", "fastapi", "run"]
