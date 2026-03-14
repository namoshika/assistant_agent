from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from agent_assistant.context import ContextSchema

SYSTEM_PROMPT = f"""
# Instruction
あなたはユーザの調べものを積極的に手助けする親身なAIエージェントです。
ユーザが調べている領域についてナレッジベースを多角的な切り口で検索し、情報を収集してください。

収集ではユーザが入力したキーワードを検索するだけでなく、収集したノートを読み、関連する事柄が書かれた
箇所に記載された wikilink に未参照のリンクが有ればさらにリンク先ノートを参照し、リンク先が調べものと
**無関係なものになるまでリンクを辿って深掘り** してください。

十分な情報を収集したと判断できたら、ユーザの調べものに関連するものに絞り、再構成された調査結果を返してください。
また、**適宜何について分かったのか、今何を調べているのかをメッセージとして出力してください**。

# Background
現在日時: {datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()}
"""  # noqa: E501


def build_graph(
    agent_name: str, llm: BaseChatModel, tools: list[BaseTool]
) -> CompiledStateGraph[Any, ContextSchema, Any, Any]:
    agentic_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=ContextSchema,
        name=agent_name,
    )
    return agentic_graph
