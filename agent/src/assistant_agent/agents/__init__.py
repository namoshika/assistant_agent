import os
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

import assistant_agent.services  # noqa: F401  登録発火
from assistant_agent import store
from assistant_agent.utils import context
from assistant_agent.utils.context import CommonContext

from . import sample


def build_agent(
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[CompiledStateGraph[Any, CommonContext, Any, Any], CommonContext]:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        lc_agent と context のペア。呼び出し元がラップ方法を選ぶ。

    """
    # コンテキスト初期化
    pg_connection_string = os.getenv("AA_PG_CONNECTION_STRING")
    assert pg_connection_string is not None
    store_conn = store.PostgresStoreConnector(pg_connection_string)
    ctx = context.ContextRegistry.build(store_conn=store_conn)

    # エージェント初期化
    lc_agent = sample.build_lc_agent(checkpointer)

    return lc_agent, ctx
