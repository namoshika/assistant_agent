import os
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr
from sqlalchemy import Engine, MetaData, create_engine, text
from sqlalchemy.orm import DeclarativeBase

from agent_assistant.loader.obsidian import VaultLoader
from agent_assistant.model import ObsidianVaultEntity
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


@pytest.fixture(scope="session")
def sa_engine():
    """SQLAlchemy Engine (セッション全体で共有)."""
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    engine = create_engine(conn_str)
    yield engine
    engine.dispose()


@pytest.fixture()
def vault_name() -> str:
    """テストごとに一意なテーブルプレフィックス."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def vault_entities(sa_engine: Engine, vault_name: str) -> Generator[type, None, None]:
    """Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """

    class _TestBase(DeclarativeBase):
        metadata = MetaData("public")

    class _TestVaultRawEntity(_TestBase, ObsidianVaultEntity):
        __tablename__ = f"{vault_name}_raw"

    _TestBase.metadata.create_all(sa_engine)
    yield _TestVaultRawEntity

    with sa_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {vault_name}_raw CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS data_{vault_name}_vectors CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS data_{vault_name}_docstore CASCADE"))
        conn.commit()


@pytest.fixture()
def obsidian_retriever(
    sa_engine: Engine, vault_name: str, vault_entities: type
) -> Generator[ObsidianLlamaRetriever, None, None]:
    """実際の PostgreSQL に接続した ObsidianLlamaRetriever.

    ENV_PG_CONNECTION_STRING と ENV_GEMINI_API_KEY 環境変数が必要。
    """
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    raw_entity = vault_entities
    retriever = ObsidianLlamaRetriever(
        sa_engine=sa_engine,
        connection_string=conn_str,
        emb_api_key=SecretStr(env_gemini_api_key),
        docstore_name=f"{vault_name}_docstore",
        vectorstore_name=f"{vault_name}_vectors",
        vault_entity=raw_entity,  # pyright: ignore[reportArgumentType]
    )
    yield retriever


@pytest.fixture()
def vault_docs() -> list[Document]:
    """VaultLoader で docs/dataset_obsidian/ から先頭 10 件を取得するフィクスチャ."""
    return VaultLoader(Path("docs/dataset_obsidian/")).load()[:10]
