from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from assistant_agent.retriever.obsidian_llama import ObsidianLlamaRetriever

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
"""  # noqa: E501


@dataclass
class ContextSchema:
    llm: BaseChatModel
    obsidian_store: ObsidianLlamaRetriever


def build_graph(
    agent_name: str, llm: BaseChatModel, tools: list[BaseTool]
) -> CompiledStateGraph[Any, ContextSchema, Any, Any]:
    """エージェントのグラフ構造を構築する.

    Args:
        agent_name: エージェントの名称。
        llm: 使用する言語モデル。
        tools: エージェントが利用可能なツールのリスト。

    Returns:
        構築された CompiledStateGraph インスタンス。

    """
    agentic_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=ContextSchema,
        name=agent_name,
    )
    return agentic_graph
