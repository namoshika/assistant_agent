import os

from mlflow.pyfunc.model import ChatAgent
from pytest_mock import MockerFixture

from assistant_agent import agents


def test_build_agent_01(mocker: MockerFixture):
    """build_agent() が ChatAgent を返せるか確認.

    観点1: 戻り値が ChatAgent インスタンスであること
    """
    # 試験準備
    mocker.patch.dict(
        os.environ,
        {
            "ENV_GEMINI_API_KEY": "dummy-key",
            "ENV_PG_CONNECTION_STRING": "postgresql://localhost/test",
        },
    )
    mocker.patch("assistant_agent.agents.entities.VaultBase.metadata.create_all")

    # 試験実施
    agent = agents.build_agent()

    # 結果検証
    # 観点1
    assert isinstance(agent, ChatAgent)
