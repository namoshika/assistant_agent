import os

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from mlflow.pyfunc.model import ChatAgent
from pydantic import SecretStr

import assistant_agent.services  # noqa: F401  登録発火
from assistant_agent import graph, tools
from assistant_agent.entities import postgres as entities
from assistant_agent.utils import context, mlflow, store_context

AGENT_NAME = "agent"


def get_model() -> BaseChatModel:
    """LLM と埋め込みモデルを生成して返す."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    assert aws_access_key_id is not None
    assert aws_secret_access_key is not None
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    assert env_gemini_api_key is not None

    # from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(
    #     model=os.environ.get("ENV_GEMINI_MODEL_ID", "gemini-3-flash-preview"),
    #     api_key=env_gemini_api_key,
    #     thinking_level="minimal",
    # )
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
    pg_connection_string = os.getenv("ENV_PG_CONNECTION_STRING")
    assert pg_connection_string is not None

    llm = get_model()
    store_ctx = store_context.PostgresStoreContext(pg_connection_string, schema_name="app")
    sa_engine = store_ctx.get_engine()
    entities.VaultBase.metadata.create_all(sa_engine)

    # エージェント初期化
    ctx = context.ContextRegistry.build(store_ctx=store_ctx)
    agent = graph.build_graph(AGENT_NAME, llm, tools.get_tools())
    agent_wrapped = mlflow.LangGraphChatAgent(agent, ctx)

    return agent_wrapped
