import os
from unittest.mock import MagicMock
from pytest_mock import MockerFixture

from agent_assistant.context import build_session, ContextSchema
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


def test_build_session_01(mocker: MockerFixture):
    """build_session() が ContextSchema を返す。

    観点1: 戻り値が ContextSchema インスタンス
    観点2: obsidian_store が ObsidianLlamaRetriever インスタンス
    """
    # 試験準備
    mocker.patch.dict(os.environ, {
        "ENV_GEMINI_API_KEY": "dummy-key",
        "ENV_PG_CONNECTION_STRING": "postgresql://localhost/test",
    })
    mocker.patch("sqlalchemy.create_engine", return_value=MagicMock())
    mocker.patch("agent_assistant.context.ObsidianVaultBase.metadata.create_all")
    mocker.patch("agent_assistant.context.ObsidianLlamaRetriever", return_value=MagicMock(spec=ObsidianLlamaRetriever))

    # 試験実施
    ctx = build_session()

    # 結果検証
    # 観点1
    assert isinstance(ctx, ContextSchema)
    # 観点2
    assert isinstance(ctx.obsidian_store, ObsidianLlamaRetriever)
