import os
from dataclasses import dataclass
from langchain_postgres import PGEngine
from sqlalchemy import create_engine
from pydantic import SecretStr
from agent_assistant.utils.chunker.text import TextChunker
from agent_assistant.retriever.obsidian import ObsidianChunkStore, ObsidianDocumentStore


@dataclass
class ContextSchema:
    obsidian_store: ObsidianDocumentStore


def build_session() -> ContextSchema:
    ENV_GEMINI_API_KEY = os.getenv("ENV_GEMINI_API_KEY")
    PG_CONNECTION_STRING = os.getenv("ENV_PG_CONNECTION_STRING")
    assert ENV_GEMINI_API_KEY is not None
    assert PG_CONNECTION_STRING is not None
    ENV_GEMINI_API_KEY = SecretStr(ENV_GEMINI_API_KEY)

    pg_engine = PGEngine.from_connection_string(url=PG_CONNECTION_STRING, pool_size=5)
    sa_engine = create_engine(PG_CONNECTION_STRING)
    obsidian_chunk = ObsidianChunkStore(pg_engine, ENV_GEMINI_API_KEY)
    obsidian_store = ObsidianDocumentStore(
        "obsidian_vault", obsidian_chunk, sa_engine, TextChunker()
    )

    return ContextSchema(obsidian_store=obsidian_store)
