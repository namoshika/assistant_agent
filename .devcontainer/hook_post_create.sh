#!/bin/sh

# Python (for JupyterLab)
curl -LsSf https://astral.sh/uv/install.sh | sh
# NodeJS (Gemini CLI, Claude Code)
curl -o- https://fnm.vercel.app/install | bash \

# Configure bashrc
cat .devcontainer/init_env.sh >> $HOME/.bashrc
. .devcontainer/init_env.sh

# Install NodeJS v24
fnm install 24 \
# Install Gemini CLI (AI Coding Agent)
npm install -g @google/gemini-cli
# Install Claude Code (AI Coding Agent)
curl -fsSL https://claude.ai/install.sh | bash
