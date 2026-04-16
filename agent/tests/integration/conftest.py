import os
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from sqlalchemy import MetaData, text
from sqlalchemy.orm import DeclarativeBase

from assistant_agent.entities import duckdb, postgres
from assistant_agent.loaders.obsidian import VaultLoader
from assistant_agent.services.vault_obsidian import VaultObsidianRetriever
from assistant_agent.utils.store_factory import DuckDBStoreContext, PostgresStoreContext


@pytest.fixture(scope="session")
def pg_cxt() -> Generator[PostgresStoreContext, None, None]:
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
def docs_obs() -> list[Document]:
    """VaultLoader で docs/dataset_obsidian/ から先頭 10 件を取得するフィクスチャ."""
    return VaultLoader(Path("docs/dataset_obsidian/")).load_data()[:10]


@pytest.fixture()
def pg_entity_obs(pg_cxt: PostgresStoreContext, vault_name: str) -> Generator[type, None, None]:
    """Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = pg_cxt.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestVaultRawEntity(_TestBase, postgres.ObsidianFields):
        __tablename__ = f"{vault_name}_raw"

    _TestBase.metadata.create_all(engine)
    yield _TestVaultRawEntity
    _TestBase.metadata.drop_all(engine)


@pytest.fixture()
def pg_retriever_obs(
    pg_cxt: PostgresStoreContext, vault_name: str, pg_entity_obs: type
) -> Generator[VaultObsidianRetriever, None, None]:
    """実際の PostgreSQL に接続した VaultObsidianRetriever.

    ENV_PG_CONNECTION_STRING と ENV_GEMINI_API_KEY 環境変数が必要。
    """
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    engine = pg_cxt.get_engine()
    retriever = VaultObsidianRetriever(
        sa_engine=engine,
        store_factory=pg_cxt,
        docstore_name=f"{vault_name}_docstore",
        vectorstore_name=f"{vault_name}_vectors",
        embed_model=GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=env_gemini_api_key,
        ),
        embed_dim=3072,
        vault_entity=pg_entity_obs,
    )
    yield retriever
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS app.data_{vault_name}_vectors CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS app.data_{vault_name}_docstore CASCADE"))
        conn.commit()



@pytest.fixture(scope="session")
def dk_cxt(tmp_path_factory: pytest.TempPathFactory) -> Generator[DuckDBStoreContext, None, None]:
    """DuckDBStoreContext（セッション全体で共有）.

    tmp_path は function スコープのため session スコープには使用不可。
    session スコープ対応の tmp_path_factory を使う。
    """
    persist_dir = tmp_path_factory.mktemp("duckdb")
    ctx = DuckDBStoreContext(persist_dir=persist_dir)
    yield ctx
    ctx.close()


@pytest.fixture()
def dk_entity_obs(dk_cxt: DuckDBStoreContext, vault_name: str) -> Generator[type, None, None]:
    """DuckDB 用 Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = dk_cxt.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData()

    class _TestVaultRawEntity(_TestBase, duckdb.ObsidianFields):
        __tablename__ = f"{vault_name}_raw"

    _TestBase.metadata.create_all(engine)
    yield _TestVaultRawEntity
    _TestBase.metadata.drop_all(engine)


@pytest.fixture()
def dk_retriever_obs(
    dk_cxt: DuckDBStoreContext, vault_name: str, dk_entity_obs: type
) -> Generator[VaultObsidianRetriever, None, None]:
    """実際の DuckDB に接続した VaultObsidianRetriever.

    ENV_GEMINI_API_KEY 環境変数が必要。
    """
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    engine = dk_cxt.get_engine()
    retriever = VaultObsidianRetriever(
        sa_engine=engine,
        store_factory=dk_cxt,
        docstore_name=f"{vault_name}_docstore",
        vectorstore_name=f"{vault_name}_vectors",
        embed_model=GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=env_gemini_api_key,
        ),
        embed_dim=3072,
        vault_entity=dk_entity_obs,
    )
    yield retriever
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS data_{vault_name}_vectors CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS data_{vault_name}_docstore CASCADE"))
        conn.commit()
