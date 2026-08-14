import uuid

import deepagents
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from assistant_agent import subagents
from assistant_agent.agents import COMMON_TOOLS


@pytest.mark.integration
async def test_web_researcher_01(llm: BaseChatModel) -> None:
    """実 LLM で web-researcher サブエージェントに Web 調査を依頼し正常終了することを確認.

    観点1: 例外なく応答が返ること
    """
    # 試験準備
    lc_agent = deepagents.create_deep_agent(
        model=llm,
        tools=COMMON_TOOLS,
        subagents=[subagents.web_researcher],
        checkpointer=InMemorySaver(),
    )
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
