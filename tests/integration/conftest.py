import os
import uuid
import pytest
from collections.abc import Generator
from langchain_core.documents import Document
from langchain_postgres import PGEngine
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from agent_assistant.retriever.obsidian import ObsidianChunkStore, ObsidianDocumentStore
from agent_assistant.utils.chunker.text import TextChunker


@pytest.fixture(scope="session")
def pg_engine():
    """langchain_postgres PGEngine (セッション全体で共有)。"""
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    return PGEngine.from_connection_string(conn_str)


@pytest.fixture(scope="session")
def sa_engine():
    """SQLAlchemy Engine (セッション全体で共有)。"""
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    engine = create_engine(conn_str)
    yield engine
    engine.dispose()


@pytest.fixture()
def vault_name() -> str:
    """テストごとに一意なテーブルプレフィックス。"""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def obsidian_store(
    pg_engine: PGEngine, sa_engine: Engine, vault_name: str
) -> Generator[ObsidianDocumentStore, None, None]:
    """実際の PostgreSQL に接続した ObsidianDocumentStore。

    ENV_GEMINI_API_KEY 環境変数が必要。テスト終了後に作成したテーブルを DROP する。
    """
    ENV_GEMINI_API_KEY = os.environ.get("ENV_GEMINI_API_KEY")
    if not ENV_GEMINI_API_KEY:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    chunk_store = ObsidianChunkStore(pg_engine, SecretStr(ENV_GEMINI_API_KEY))
    store = ObsidianDocumentStore(
        store_name=vault_name,
        chunk_store=chunk_store,
        sa_engine=sa_engine,
        chunker=TextChunker(chunk_size=128),
    )
    store.connect()
    yield store

    with sa_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {vault_name}_raw CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS {vault_name}_chunks CASCADE"))
        conn.commit()


def make_docs() -> list[Document]:
    """テスト用 Document リストを生成するヘルパー。"""
    return [
        Document(
            page_content="LangChain は LLM アプリケーション構築フレームワークである。",
            metadata={"path": "langchain.md", "tags": ["ai", "framework"]},
        ),
        Document(
            page_content="PostgreSQL は高性能なオープンソースデータベースである。",
            metadata={"path": "postgres.md", "tags": ["database"]},
        ),
    ]
