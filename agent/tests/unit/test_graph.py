from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from assistant_agent.graph import ContextSchema, build_graph


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> "_FakeChatModel":
        return self


def test_build_graph_01():
    """build_graph() がエージェントを返し、呼び出しで LLM と tool が実行できるか確認.

    観点1: 戻り値が CompiledStateGraph インスタンス
    観点2: インスタンスを invoke すると LLM が呼ばれ AIMessage が生成される
    """
    fake_llm = _FakeChatModel(messages=iter([AIMessage(content="hello")]))

    # 試験実施
    agent = build_graph("test-agent", fake_llm, [])
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="こんにちは?"),
            ]
        },
        context=ContextSchema(llm=MagicMock(), obsidian_store=MagicMock()),
    )
    msg = next(m for m in result["messages"])

    # 観点1
    assert isinstance(agent, CompiledStateGraph)
    # 観点2
    assert msg is not None
