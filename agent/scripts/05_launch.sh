#!/bin/bash
# 複数の solbot エージェントを同時起動し、Ctrl+C で一括停止するスクリプト。
# トークン等の秘密情報は直書きせず、リポジトリルートの .env から読み込む。
#
# .env に以下を定義しておくこと:
#   AA_DISCORD_BOT_TOKEN_1=...
#   AA_DISCORD_BOT_TOKEN_2=...

set -eu
cd "$(dirname "$0")/.."

set -a
. ./.env
set +a

pids=()

stop_all() {
    echo -e "\nStopping all agents..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}
trap stop_all INT TERM

AA_DISCORD_BOT_TOKEN="$AA_DISCORD_BOT_TOKEN_1" uv run solbot --agent rune & pids+=("$!")
AA_DISCORD_BOT_TOKEN="$AA_DISCORD_BOT_TOKEN_2" uv run solbot --agent holo & pids+=("$!")

wait
