import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import langchain.agents
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from assistant_agent.tools import obsidian, sample
from assistant_agent.utils.context import CommonContext

SYSTEM_PROMPT = f"""
# Instruction
あなたはユーザの調べものを積極的に手助けする親身なAIエージェントです。
ユーザが問い掛けに対してナレッジベースから関連する事柄を多角的な切り口で検索し、情報を収集してください。

ユーザが希望する場合、
収集ではユーザが入力したキーワードを検索するだけでなく、収集したノートを読み、関連する事柄が書かれた
箇所に記載された wikilink に未参照のリンクが有ればさらにリンク先ノートを参照し、深掘りしてください。

十分な情報を収集したと判断できたら、ユーザの調べものに関連するものに絞り、再構成された調査結果を返してください。
また、**適宜何について分かったのか、今何を調べているのかをメッセージとして出力してください**。

# Background
現在日時: {datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()}
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
        model="qwen.qwen3-235b-a22b-2507-v1:0",
        aws_access_key_id=SecretStr(aws_access_key_id),
        aws_secret_access_key=SecretStr(aws_secret_access_key),
        region_name=aws_default_region,
    )
    return langchain.agents.create_agent(
        model=llm,
        tools=[
            sample.get_weather,
            sample.sample_search,
            obsidian.obsidian_vault_search,
            obsidian.obsidian_vault_get,
        ],
        system_prompt=SYSTEM_PROMPT,
        context_schema=CommonContext,
        name="agent",
        checkpointer=checkpointer,
    )
