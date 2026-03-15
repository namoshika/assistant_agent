import os
import sqlalchemy
from dataclasses import dataclass
from pydantic import SecretStr
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever
from agent_assistant.model import (
    ObsidianVaultBase,
    ObsidianVaultRawEntity,
)


@dataclass
class ContextSchema:
    obsidian_store: ObsidianLlamaRetriever


def build_session() -> ContextSchema:
    ENV_VAULT_NAME = os.getenv("ENV_VAULT_NAME", "obsidian_vault")
    ENV_GEMINI_API_KEY = os.getenv("ENV_GEMINI_API_KEY")
    PG_CONNECTION_STRING = os.getenv("ENV_PG_CONNECTION_STRING")
    assert ENV_GEMINI_API_KEY is not None
    assert PG_CONNECTION_STRING is not None
    ENV_GEMINI_API_KEY = SecretStr(ENV_GEMINI_API_KEY)

    sa_engine = sqlalchemy.create_engine(PG_CONNECTION_STRING)

    ObsidianVaultBase.metadata.create_all(sa_engine)
    obsidian_store = ObsidianLlamaRetriever(
        sa_engine=sa_engine,
        connection_string=PG_CONNECTION_STRING,
        emb_api_key=ENV_GEMINI_API_KEY,
        docstore_name=f"{ENV_VAULT_NAME}_docs",
        vectorstore_name=f"{ENV_VAULT_NAME}_vectors",
        vault_entity=ObsidianVaultRawEntity,
    )

    return ContextSchema(obsidian_store=obsidian_store)
