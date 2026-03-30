import os
from dataclasses import dataclass

import sqlalchemy
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from agent_assistant.model import (
    ObsidianVaultBase,
    ObsidianVaultRawEntity,
)
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


@dataclass
class ContextSchema:
    obsidian_store: ObsidianLlamaRetriever


def build_session() -> ContextSchema:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ObsidianLlamaRetriever を含む ContextSchema インスタンス。

    """
    env_vault_name = os.getenv("ENV_VAULT_NAME", "obsidian_vault")
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    pg_connection_string = os.getenv("DEV_PG_CONNECTION_STRING")
    assert env_gemini_api_key is not None
    assert pg_connection_string is not None

    sa_engine = sqlalchemy.create_engine(pg_connection_string)
    embed_model = GoogleGenAIEmbedding(
        model="gemini-embedding-001",
        api_key=env_gemini_api_key,
    )

    ObsidianVaultBase.metadata.create_all(sa_engine)
    obsidian_store = ObsidianLlamaRetriever(
        sa_engine=sa_engine,
        connection_string=pg_connection_string,
        docstore_name=f"{env_vault_name}_docs",
        vectorstore_name=f"{env_vault_name}_vectors",
        embed_model=embed_model,
        vault_entity=ObsidianVaultRawEntity,
    )

    return ContextSchema(obsidian_store=obsidian_store)
