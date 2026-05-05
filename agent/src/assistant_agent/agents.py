import os

from langchain_core.language_models import BaseChatModel
from mlflow.pyfunc.model import ChatAgent

import assistant_agent.services  # noqa: F401  登録発火
from assistant_agent import graph, store, tools
from assistant_agent.utils import context, mlflow

AGENT_NAME = "agent"


def get_model() -> BaseChatModel:
    """LLM と埋め込みモデルを生成して返す."""
    from langchain_aws import ChatBedrockConverse
    from pydantic import SecretStr

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    assert aws_access_key_id is not None
    assert aws_secret_access_key is not None

    llm = ChatBedrockConverse(
        model="qwen.qwen3-235b-a22b-2507-v1:0",
        # model="minimax.minimax-m2.5",
        aws_access_key_id=SecretStr(aws_access_key_id),
        aws_secret_access_key=SecretStr(aws_secret_access_key),
        region_name=aws_default_region,
    )
    return llm


def build_agent() -> ChatAgent:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ChatAgent インスタンス。

    """
    # コンテキスト初期化
    pg_connection_string = os.getenv("ENV_PG_CONNECTION_STRING")
    assert pg_connection_string is not None
    store_conn = store.PostgresStoreConnector(pg_connection_string, schema_name="app")
    ctx = context.ContextRegistry.build(store_conn=store_conn)

    # エージェント初期化
    agent = graph.build_graph(AGENT_NAME, get_model(), tools.get_tools())
    agent_wrapped = mlflow.LangGraphChatAgent(agent, ctx)

    return agent_wrapped
