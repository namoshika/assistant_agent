FROM mcr.microsoft.com/devcontainers/base:noble

USER vscode
WORKDIR /workspace
COPY --chown=ubuntu:ubuntu build_container/ /workspace/build_container/
RUN ["/bin/bash", "build_container/init_env.sh"]

# OpenAI Compatible API endpoint port
EXPOSE 8000/tcp
