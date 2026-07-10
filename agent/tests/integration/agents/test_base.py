import pytest
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from assistant_agent import agents


@pytest.mark.integration
def test_build_agent_01():
    """build_agent() が ChatAgent を生成してレスポンスを返せるか確認.

    観点1: 戻り値が ChatAgent インスタンスであること
    観点2: predict するとレスポンスが返ること
    """
    # 試験実施
    agent = agents.build_agent()

    # 結果検証
    # 観点1
    assert isinstance(agent, ChatAgent)

    # 観点2
    response = agent.predict(messages=[ChatAgentMessage(role="user", content="こんにちは")])
    assert isinstance(response, ChatAgentResponse)
    assert len(response.messages) >= 1
    assert response.messages[0].content
