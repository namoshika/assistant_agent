import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from assistant_agent import agents


@pytest.mark.integration
async def test_invoke_01():
    """実 LLM で invoke() が意味のある応答を返し、履歴が引き継がれるか確認.

    観点1: 実 LLM で単発の invoke() が意味のある応答を返すこと
    観点2: InMemorySaver を渡すと複数回の invoke() で会話文脈が引き継がれること
    """
    # 試験準備
    agent = agents.SampleAgent(checkpointer=InMemorySaver(), thread_id="integration-test")
    config: RunnableConfig = {"configurable": {"thread_id": "integration-test"}}

    # 試験実施
    result_1st = await agent.invoke(
        {"messages": [HumanMessage(content="私の名前はテスト太郎です。覚えてください。")]},
        config=config,
    )
    result_2nd = await agent.invoke(
        {"messages": [HumanMessage(content="私の名前は何ですか？")]}, config=config
    )

    # 結果検証
    # 観点1
    assert result_1st.value["messages"][-1].content
    # 観点2
    assert "テスト太郎" in result_2nd.value["messages"][-1].content
