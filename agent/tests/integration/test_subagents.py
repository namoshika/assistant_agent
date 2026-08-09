import uuid

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from assistant_agent.agents import sample


@pytest.mark.integration
async def test_web_researcher_01(llm: BaseChatModel) -> None:
    """実 LLM で web-researcher サブエージェントに Web 調査を依頼し正常終了することを確認.

    観点1: 例外なく応答が返ること
    """
    # 試験準備
    lc_agent = sample.build_lc_agent(InMemorySaver(), InMemoryStore(), llm, "sample")
    config: RunnableConfig = {"configurable": {"thread_id": f"test-agent:{uuid.uuid7()}"}}

    # 試験実施
    result = await lc_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="web-researcher サブエージェントに、"
                    "本日の東京の天気を Web で調べてもらってください。"
                )
            ]
        },
        config=config,
    )

    # 結果検証
    # 観点1
    assert result["messages"][-1].content
