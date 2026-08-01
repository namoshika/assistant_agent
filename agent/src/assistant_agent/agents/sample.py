import os
from typing import Any

import langchain.agents
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from assistant_agent.tools import discord, dispatcher, obsidian, sample
from assistant_agent.utils.context import CommonContext

SYSTEM_PROMPT = """
# Instruction
あなたは親身なAIエージェントです。
ユーザの必要に対してツールを使いながら応えてください。

# Guideline
エージェントはいくつかのチャンネルから外界のイベントを入力されます。イベントから外界の状態を理解し、行動してください。
ユーザのメッセージもイベントとしてチャネルから受信します。必要に応じて応答してください。

## Discord Channel
ユーザからメッセージが来ます。メンションでの指示や、エージェント自身が応えられる事柄には応答してください。
応答は受信した Discord サーバーの同じチャンネルへ送信してください。

## Dispatcher Channel
エージェント自身が過去に送信予約したメッセージが入力されます。

## Discord Post

- 新着受信時
  - 直近投稿を 20 件取得し、直近の文脈を把握。必要の応じて応答を返す
- 投稿時 (通常の投稿をする場合): メンションなしで投稿
- 投稿時 (特定の投稿やユーザへリアクションする場合)
  - 対象の投稿が直近10件以内の投稿の場合: メンションなしで投稿
  - 対象の投稿が直近10件より前の投稿の場合: リプライを使用
  - 対象の相手を指名した投稿の場合: メンションを使用

## Dispatcher Tool
ユーザ要求へ応えるためにエージェント自身が後の時刻に自律動作したい場合は
Dispatcher を使用し送信予約を入れてください。Dispatcher Channel を通して起動されます。

- 指定日時に送信予約する場合
  - 予約前に現在時刻を取得して未来の日付あることを確認してから行ってください。
    過去の日付は指定できません。

"""


def build_lc_agent(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[Any, CommonContext, Any, Any]:
    """LLM・ツール・checkpointer を束ねたグラフを構築する."""
    from langchain_aws import ChatBedrockConverse
    from pydantic import SecretStr

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    assert aws_access_key_id is not None
    assert aws_secret_access_key is not None

    llm = ChatBedrockConverse(
        model="global.anthropic.claude-sonnet-5",
        aws_access_key_id=SecretStr(aws_access_key_id),
        aws_secret_access_key=SecretStr(aws_secret_access_key),
        region_name=aws_default_region,
    )
    return langchain.agents.create_agent(
        model=llm,
        tools=[
            sample.get_weather,
            sample.get_datetime_now,
            dispatcher.dispatcher_invoke_at,
            dispatcher.dispatcher_invoke_delay,
            dispatcher.dispatcher_cancel,
            dispatcher.dispatcher_list,
            sample.sample_search,
            obsidian.obsidian_vault_search,
            obsidian.obsidian_vault_get,
            discord.discord_get_messages,
            discord.discord_send_message,
            discord.discord_mention_user,
            discord.discord_reply_message,
        ],
        system_prompt=SYSTEM_PROMPT,
        context_schema=CommonContext,
        name="agent",
        checkpointer=checkpointer,
    )
