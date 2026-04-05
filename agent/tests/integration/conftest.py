import os
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from langchain_core.documents import Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from sqlalchemy import MetaData, text
from sqlalchemy.orm import DeclarativeBase

from agent_assistant.entities import ObsidianVaultEntity
from agent_assistant.loader.obsidian import VaultLoader
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever
from agent_assistant.utils.store_factory import PostgresStoreContext


@pytest.fixture(scope="session")
def factory() -> Generator[PostgresStoreContext, None, None]:
    """PostgresStoreContext (セッション全体で共有)."""
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    f = PostgresStoreContext(conn_str, schema_name="app")
    yield f
    f.get_engine().dispose()


@pytest.fixture()
def vault_name() -> str:
    """テストごとに一意なテーブルプレフィックス."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def vault_entities(factory: PostgresStoreContext, vault_name: str) -> Generator[type, None, None]:
    """Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = factory.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestVaultRawEntity(_TestBase, ObsidianVaultEntity):
        __tablename__ = f"{vault_name}_raw"

    _TestBase.metadata.create_all(engine)
    yield _TestVaultRawEntity
    _TestBase.metadata.drop_all(engine)

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS app.data_{vault_name}_vectors CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS app.data_{vault_name}_docstore CASCADE"))
        conn.commit()


@pytest.fixture()
def obsidian_retriever(
    factory: PostgresStoreContext, vault_name: str, vault_entities: type
) -> Generator[ObsidianLlamaRetriever, None, None]:
    """実際の PostgreSQL に接続した ObsidianLlamaRetriever.

    ENV_PG_CONNECTION_STRING と ENV_GEMINI_API_KEY 環境変数が必要。
    """
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    retriever = ObsidianLlamaRetriever(
        sa_engine=factory.get_engine(),
        store_factory=factory,
        docstore_name=f"{vault_name}_docstore",
        vectorstore_name=f"{vault_name}_vectors",
        embed_model=GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=env_gemini_api_key,
        ),
        embed_dim=3072,
        vault_entity=vault_entities,  # pyright: ignore[reportArgumentType]
    )
    yield retriever


@pytest.fixture()
def vault_docs() -> list[Document]:
    """VaultLoader で docs/dataset_obsidian/ から先頭 10 件を取得するフィクスチャ."""
    return VaultLoader(Path("docs/dataset_obsidian/")).load()[:10]
