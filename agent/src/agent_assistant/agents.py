import os

import mlflow
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from mlflow.pyfunc.model import ChatAgent
from pydantic import SecretStr

from agent_assistant import entities, graph, tools
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever
from agent_assistant.utils.mlflow import LangGraphChatAgent
from agent_assistant.utils.store_factory import PostgresStoreContext

AGENT_NAME = "agent"
mlflow.set_experiment("agent-rag")
mlflow.autolog()


def get_model() -> tuple[BaseChatModel, BaseEmbedding]:
    """LLM と埋め込みモデルを生成して返す.

    Returns:
        (llm, emb) のタプル。

    """
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
    emb = GoogleGenAIEmbedding(
        model="gemini-embedding-001",
        api_key=env_gemini_api_key,
    )
    return llm, emb


def build_agent() -> ChatAgent:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ChatAgent インスタンス。

    """
    env_vault_name = os.getenv("ENV_VAULT_NAME", "obsidian_vault")
    pg_connection_string = os.getenv("ENV_PG_CONNECTION_STRING")
    assert pg_connection_string is not None

    llm, emb = get_model()

    factory = PostgresStoreContext(pg_connection_string, schema_name="app")
    sa_engine = factory.get_engine()
    entities.ObsidianVaultBase.metadata.create_all(sa_engine)
    obsidian_store = ObsidianLlamaRetriever(
        sa_engine=sa_engine,
        store_factory=factory,
        docstore_name=f"{env_vault_name}_docs",
        vectorstore_name=f"{env_vault_name}_vectors",
        embed_model=emb,
        embed_dim=3072,
    )

    ctx = graph.ContextSchema(llm=llm, obsidian_store=obsidian_store)
    agent = graph.build_graph(AGENT_NAME, ctx.llm, tools.get_tools())
    agent_wrapped = LangGraphChatAgent(agent, ctx)  # pyright: ignore[reportArgumentType]

    return agent_wrapped
