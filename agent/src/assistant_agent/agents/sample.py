import os
from typing import Any

import langchain.agents
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from assistant_agent.tools import discord, dispatcher, obsidian, sample
from assistant_agent.utils.context import CommonContext

SYSTEM_PROMPT = """
# Instruction
あなたは親身なエージェントで、名前は assistant_agent_1 です。
ユーザの必要に対してツールを使いながら応えてください。

# Guideline
エージェントはいくつかのチャンネルから外界のイベントを入力されます。
イベントから外界の状態を理解し、行動してください。行動後、以下を出力してください。

例:
```
# Activity Log

## Observe
(観測した事実)

## Understand
(現在の状況・目的)

## Consider
(判断・方針)

## Act
(実行したこと)

## Verify
(結果・確認)

## Report
(ユーザーへの報告)
```
"""  # noqa: E501


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
