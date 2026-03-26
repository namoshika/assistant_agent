import importlib
import os
import sys

import pytest
from langchain_core.documents import Document
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from pytest_mock import MockerFixture
from sqlalchemy import Engine

from agent_assistant.loader.obsidian import PgVault
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


@pytest.mark.integration
def test_agent_01(
    mocker: MockerFixture,
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
    vault_name: str,
) -> None:
    """適切に初期化されたエージェントが mlflow へ登録される.

    観点1: 登録されたエージェントが ChatAgent のインスタンスである
    観点2: predict() が ChatAgentResponse を返す
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        pytest.fail("AWS 認証情報が未設定")

    # 試験準備
    raw_entity = vault_entities
    PgVault.sync(vault_docs, sa_engine, raw_entity)
    obsidian_retriever.sync_chunks()
    sys.modules.pop("agent_assistant.agent", None)
    mock_set_model = mocker.patch("mlflow.models.set_model")
    mocker.patch.dict(os.environ, {"ENV_VAULT_NAME": vault_name})

    # 試験実施
    importlib.import_module("agent_assistant.agent")
    agent_wrapped = mock_set_model.call_args.args[0]

    # 観点1
    assert isinstance(agent_wrapped, ChatAgent)
    result = agent_wrapped.predict(
        messages=[ChatAgentMessage(role="user", content="東京の天気は?")]
    )

    # 観点2
    assert isinstance(result, ChatAgentResponse)
