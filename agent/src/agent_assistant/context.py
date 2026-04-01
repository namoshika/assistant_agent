import os
from dataclasses import dataclass

import sqlalchemy
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pydantic import SecretStr

from agent_assistant.model import (
    ObsidianVaultBase,
    ObsidianVaultRawEntity,
)
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


@dataclass
class ContextSchema:
    llm: BaseChatModel
    obsidian_store: ObsidianLlamaRetriever


def build_session() -> ContextSchema:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ContextSchema インスタンス。

    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    assert aws_access_key_id is not None
    assert aws_secret_access_key is not None
    aws_access_key_id = SecretStr(aws_access_key_id)
    aws_secret_access_key = SecretStr(aws_secret_access_key)

    env_vault_name = os.getenv("ENV_VAULT_NAME", "obsidian_vault")
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    pg_connection_string = os.getenv("DEV_PG_CONNECTION_STRING")
    assert env_gemini_api_key is not None
    assert pg_connection_string is not None

    # from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(
    #     model=os.environ.get("ENV_GEMINI_MODEL_ID", "gemini-3-flash-preview"),
    #     api_key=env_gemini_api_key,
    #     thinking_level="minimal",
    # )
    llm = ChatBedrockConverse(
        model="qwen.qwen3-235b-a22b-2507-v1:0",
        # model="minimax.minimax-m2.5",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_default_region,
    )
    embed_model = GoogleGenAIEmbedding(
        model="gemini-embedding-001",
        api_key=env_gemini_api_key,
    )

    sa_engine = sqlalchemy.create_engine(pg_connection_string)
    ObsidianVaultBase.metadata.create_all(sa_engine)
    obsidian_store = ObsidianLlamaRetriever(
        sa_engine=sa_engine,
        connection_string=pg_connection_string,
        docstore_name=f"{env_vault_name}_docs",
        vectorstore_name=f"{env_vault_name}_vectors",
        embed_model=embed_model,
        vault_entity=ObsidianVaultRawEntity,
    )

    return ContextSchema(llm=llm, obsidian_store=obsidian_store)
