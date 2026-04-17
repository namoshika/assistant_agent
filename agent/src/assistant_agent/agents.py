import os

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from mlflow.pyfunc.model import ChatAgent
from pydantic import SecretStr

from assistant_agent import graph, tools
from assistant_agent.entities import postgres as entities
from assistant_agent.services import VaultObsidianRetriever
from assistant_agent.utils import absclass, mlflow
from assistant_agent.utils.store_context import PostgresStoreContext

AGENT_NAME = "agent"


def get_model() -> tuple[BaseChatModel, BaseEmbedding]:
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
    emb = GoogleGenAIEmbedding(
        model="gemini-embedding-001",
        api_key=env_gemini_api_key,
    )
    return llm, emb


def get_retriever_obsidian(
    store_ctx: absclass.StoreContext, emb: BaseEmbedding
) -> VaultObsidianRetriever:
    """Obsidian レトリーバーを生成して返す."""
    retriever = VaultObsidianRetriever(
        docstore_name="obsidian_vault_docs",
        vectorstore_name="obsidian_vault_vectors",
        store_context=store_ctx,
        transformations=[
            SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                paragraph_separator="\n\n",
            ),
        ],
        embed_model=emb,
        embed_dim=3072,
        vault_entity=entities.ObsidianEntity,
    )
    return retriever


def build_agent() -> ChatAgent:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ChatAgent インスタンス。

    """
    pg_connection_string = os.getenv("ENV_PG_CONNECTION_STRING")
    assert pg_connection_string is not None

    llm, emb = get_model()
    store_ctx = PostgresStoreContext(pg_connection_string, schema_name="app")
    sa_engine = store_ctx.get_engine()
    entities.VaultBase.metadata.create_all(sa_engine)

    # エージェント初期化
    ret_obsidian = get_retriever_obsidian(store_ctx, emb)
    ctx = graph.ContextSchema(llm=llm, obsidian_retriever=ret_obsidian)
    agent = graph.build_graph(AGENT_NAME, ctx.llm, tools.get_tools())
    agent_wrapped = mlflow.LangGraphChatAgent(agent, ctx)  # pyright: ignore[reportArgumentType]

    return agent_wrapped
