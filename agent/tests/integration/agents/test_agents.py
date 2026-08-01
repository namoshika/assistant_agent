import pytest
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from assistant_agent import agents
from assistant_agent.utils.mlflow import LangGraphChatAgent


@pytest.mark.integration
async def test_build_agent_01():
    """build_agent() の戻り値から LangGraphChatAgent を構築しレスポンスを返せるか確認.

    観点1: predict するとレスポンスが返ること
    """
    # 試験準備
    lc_agent, ctx = agents.build_agent()
    agent = LangGraphChatAgent(lc_agent, ctx)

    # 試験実施
    response = await agent.predict_async(
        messages=[ChatAgentMessage(role="user", content="こんにちは")]
    )

    # 結果検証
    # 観点1
    assert isinstance(response, ChatAgentResponse)
    assert len(response.messages) >= 1
    assert response.messages[0].content
