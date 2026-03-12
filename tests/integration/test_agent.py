import importlib
import os
import sys
import pytest
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.types.responses_helpers import Message
from pytest_mock import MockerFixture

from agent_assistant.retriever.obsidian import ObsidianDocumentStore
from tests.integration.conftest import make_docs


@pytest.mark.integration
def test_agent_01(
    mocker: MockerFixture, obsidian_store: ObsidianDocumentStore, vault_name: str
) -> None:
    """agent.py インポートにより適切に初期化されたエージェントが mlflow へ登録される。

    観点1: 登録されたエージェントが ResponsesAgent のインスタンスである
    観点2: predict() が ResponsesAgentResponse を返す
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        pytest.fail("AWS 認証情報が未設定")

    # 試験準備
    obsidian_store.connect()
    obsidian_store.import_documents(make_docs())
    sys.modules.pop("agent_assistant.agent", None)
    mock_set_model = mocker.patch("mlflow.models.set_model")
    mocker.patch.dict(os.environ, {"ENV_VAULT_NAME": vault_name})

    # 試験実施
    importlib.import_module("agent_assistant.agent")
    agent_wrapped = mock_set_model.call_args.args[0]

    # 観点1
    assert isinstance(agent_wrapped, ResponsesAgent)
    result = agent_wrapped.predict(
        ResponsesAgentRequest(input=[Message(role="user", content="東京の天気は?")])
    )

    # 観点2
    assert isinstance(result, ResponsesAgentResponse)
