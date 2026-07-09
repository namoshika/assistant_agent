import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from sqlalchemy import MetaData, text
from sqlalchemy.orm import DeclarativeBase

from assistant_agent.entities import postgres
from assistant_agent.entities.postgres import ChunkFields
from assistant_agent.loaders import ObsidianLoader
from assistant_agent.services import VaultObsidianRetriever, VaultSampleRetriever
from assistant_agent.store import PostgresStoreConnector


@pytest.fixture()
async def pg_conn() -> AsyncIterator[PostgresStoreConnector]:
    """PostgresStoreConnector (テストごとに生成).

    AsyncEngine は asyncpg 接続がテスト単位のイベントループに紐づくため、
    session scope で使い回すとテストをまたいだ際に
    InterfaceError（イベントループ不整合）が発生する。そのため function scope とする。
    """
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    f = PostgresStoreConnector(conn_str)
    yield f
    await f.get_engine().dispose()


@pytest.fixture()
def vault_name() -> str:
    """テストごとに一意なテーブルプレフィックス."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
async def pg_entity_chk(pg_conn: PostgresStoreConnector, vault_name: str) -> AsyncIterator[type]:
    """一意なテーブル名を持つチャンク Entity を生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """

    class _TestChunkBase(DeclarativeBase):
        metadata = MetaData("app")

    class _TestChunkEntity(_TestChunkBase, ChunkFields):
        __tablename__ = f"{vault_name}_vectors"

    engine = pg_conn.get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.commit()
    async with engine.begin() as conn:
        await conn.run_sync(_TestChunkBase.metadata.create_all)
    yield _TestChunkEntity
    async with engine.begin() as conn:
        await conn.run_sync(_TestChunkBase.metadata.drop_all)


@pytest.fixture()
def docs_obs() -> list[Document]:
    """ObsidianReader で docs/dataset_obsidian/ から先頭 10 件を取得するフィクスチャ."""
    return ObsidianLoader(Path("docs/dataset_obsidian/")).load()[:10]


@pytest.fixture()
async def pg_entity_obs(pg_conn: PostgresStoreConnector, vault_name: str) -> AsyncIterator[type]:
    """Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = pg_conn.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestVaultRawEntity(_TestBase, postgres.ObsidianFields):
        __tablename__ = f"{vault_name}_raw"

    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.create_all)
    yield _TestVaultRawEntity
    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.drop_all)


@pytest.fixture()
def pg_retriever_obs(
    pg_conn: PostgresStoreConnector, pg_entity_obs: type, pg_entity_chk: type
) -> VaultObsidianRetriever:
    """実際の PostgreSQL に接続した VaultObsidianRetriever.

    ENV_PG_CONNECTION_STRING と ENV_GEMINI_API_KEY 環境変数が必要。
    """
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    return VaultObsidianRetriever(
        store_conn=pg_conn,
        chunk_entity=pg_entity_chk,
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(env_gemini_api_key),
        ),
        splitter=RecursiveCharacterTextSplitter(),
        vault_entity=pg_entity_obs,
    )


@pytest.fixture()
def docs_smpl() -> list[Document]:
    """結合テスト用 サンプル ドキュメントリスト."""
    return [
        Document(
            page_content=f"ウェブサイトコンテンツ {i}",
            id=f"website-doc-{i:02d}",
            metadata={"file_path": f"https://example.com/page{i}"},
        )
        for i in range(3)
    ]


@pytest.fixture()
async def pg_entity_smpl(pg_conn: PostgresStoreConnector, vault_name: str) -> AsyncIterator[type]:
    """サンプル用 Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = pg_conn.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestSampleEntity(_TestBase, postgres.DocumentFields):
        __tablename__ = f"{vault_name}_raw"

    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.create_all)
    yield _TestSampleEntity
    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.drop_all)


@pytest.fixture()
def pg_retriever_smpl(
    pg_conn: PostgresStoreConnector, pg_entity_smpl: type, pg_entity_chk: type
) -> VaultSampleRetriever:
    """実際の PostgreSQL に接続した VaultSampleRetriever."""
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    return VaultSampleRetriever(
        chunk_entity=pg_entity_chk,
        store_conn=pg_conn,
        splitter=RecursiveCharacterTextSplitter(),
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(env_gemini_api_key),
        ),
        vault_entity=pg_entity_smpl,
    )
