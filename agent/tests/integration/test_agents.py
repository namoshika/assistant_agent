import pytest
from langchain_core.language_models import BaseChatModel
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from assistant_agent import agents


@pytest.mark.integration
def test_get_model_01():
    """get_model() が BaseChatModel を生成してレスポンスを返せるか確認.

    観点1: 戻り値が BaseChatModel インスタンスであること
    観点2: invoke するとレスポンスが返ること
    """
    # 試験実施
    model = agents.get_model()

    # 結果検証
    # 観点1
    assert isinstance(model, BaseChatModel)

    # 観点2
    response = model.invoke("こんにちは")
    assert response.content


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
