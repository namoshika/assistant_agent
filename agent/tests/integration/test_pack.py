import importlib
import os
import sys

import pytest
from langchain_core.documents import Document
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from pytest_mock import MockerFixture

from agent_assistant.loader.obsidian import VaultDb
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever
from agent_assistant.utils.store_factory import PostgresStoreContext


@pytest.mark.integration
def test_agent_01(
    mocker: MockerFixture,
    pg_cxt: PostgresStoreContext,
    pg_obsidian_retriever: ObsidianLlamaRetriever,
    pg_entity: type,
    vault_docs: list[Document],
    vault_name: str,
) -> None:
    """適切に初期化されたエージェントが mlflow へ登録されるか確認.

    観点1: 登録されたエージェントが ChatAgent のインスタンスである
    観点2: 登録されたエージェントの predict() が正常動作すること
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        pytest.fail("AWS 認証情報が未設定")

    # 試験準備
    raw_entity = pg_entity
    VaultDb.sync(vault_docs, pg_cxt.get_engine(), raw_entity)
    pg_obsidian_retriever.sync_chunks()
    sys.modules.pop("agent_assistant.pack", None)
    mock_set_model = mocker.patch("mlflow.models.set_model")
    mocker.patch.dict(os.environ, {"ENV_VAULT_NAME": vault_name})

    # 試験実施
    importlib.import_module("agent_assistant.pack")
    agent_wrapped = mock_set_model.call_args.args[0]

    # 観点1
    assert isinstance(agent_wrapped, ChatAgent)

    # 観点2
    result = agent_wrapped.predict(
        messages=[ChatAgentMessage(role="user", content="東京の天気は?")]
    )
    assert isinstance(result, ChatAgentResponse)
