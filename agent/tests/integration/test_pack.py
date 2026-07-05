import importlib
import os
import sys

import pytest
from langchain_core.documents import Document
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from pytest_mock import MockerFixture

from assistant_agent.entities.base import VaultUtils
from assistant_agent.services import VaultObsidianRetriever
from assistant_agent.store import PostgresStoreConnector


@pytest.mark.integration
def test_agent_01(
    mocker: MockerFixture,
    pg_conn: PostgresStoreConnector,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type,
    docs_obs: list[Document],
) -> None:
    """適切に初期化されたエージェントが mlflow へ登録されるか確認.

    観点1: 登録されたエージェントが ChatAgent のインスタンスである
    観点2: 登録されたエージェントの predict() が正常動作すること
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        pytest.fail("AWS 認証情報が未設定")

    # 試験準備
    raw_entity = pg_entity_obs
    VaultUtils.sync_docs(docs_obs, pg_conn.get_engine(), raw_entity)
    pg_retriever_obs.sync_chunks()
    sys.modules.pop("assistant_agent.pack", None)
    mock_set_model = mocker.patch("mlflow.models.set_model")

    # 試験実施
    importlib.import_module("assistant_agent.pack")
    agent_wrapped = mock_set_model.call_args.args[0]

    # 観点1
    assert isinstance(agent_wrapped, ChatAgent)

    # 観点2
    result = agent_wrapped.predict(
        messages=[ChatAgentMessage(role="user", content="東京の天気は?")]
    )
    assert isinstance(result, ChatAgentResponse)
