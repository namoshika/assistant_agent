import os

import assistant_agent.services  # noqa: F401  登録発火
from assistant_agent import store
from assistant_agent.utils import context, mlflow

from .base import BaseAgent
from .sample import SampleAgent


def build_agent() -> mlflow.LangGraphChatAgent:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ChatAgent インスタンス。

    """
    # コンテキスト初期化
    pg_connection_string = os.getenv("ENV_PG_CONNECTION_STRING")
    assert pg_connection_string is not None
    store_conn = store.PostgresStoreConnector(pg_connection_string)
    ctx = context.ContextRegistry.build(store_conn=store_conn)

    # エージェント初期化
    agent = SampleAgent(None)
    agent_wrapped = mlflow.LangGraphChatAgent(agent.lc_agent, ctx)

    return agent_wrapped


__all__ = [
    "BaseAgent",
    "SampleAgent",
]
