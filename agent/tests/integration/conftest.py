import os
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from sqlalchemy import MetaData, text
from sqlalchemy.orm import DeclarativeBase

from assistant_agent.entities import postgres
from assistant_agent.entities.postgres import ChunkFields
from assistant_agent.loaders import ObsidianLoader
from assistant_agent.services import VaultObsidianRetriever, VaultSampleRetriever
from assistant_agent.store import PostgresStoreConnector


@pytest.fixture(scope="session", autouse=True)
def setup() -> Iterator[None]:
    """Mlflow."""
    import mlflow

    # トレース用設定（agent_server.py と同様。logger.exception() から mlflow のトレースIDを
    # 参照できるようにするため、常駐プロセスでも autolog を有効化する）
    mlflow.set_experiment(experiment_name="agent-rag")
    mlflow.openai.autolog()  # pyright: ignore[reportPrivateImportUsage]
    mlflow.gemini.autolog()  # pyright: ignore[reportPrivateImportUsage]
    mlflow.langchain.autolog(run_tracer_inline=True)  # pyright: ignore[reportPrivateImportUsage]

    session_id = f"pytest-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    with mlflow.start_run(run_name=session_id):
        yield


@pytest.fixture()
async def pg_conn() -> AsyncIterator[PostgresStoreConnector]:
    """PostgresStoreConnector (テストごとに生成).

    AsyncEngine は asyncpg 接続がテスト単位のイベントループに紐づくため、
    session scope で使い回すとテストをまたいだ際に
    InterfaceError（イベントループ不整合）が発生する。そのため function scope とする。
    """
    conn_str = os.environ.get("AA_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("AA_PG_CONNECTION_STRING が未設定のため失敗")
    f = PostgresStoreConnector(conn_str)
    yield f
    await f.get_engine().dispose()


@pytest.fixture()
def test_id() -> str:
    """テストごとに一意な識別子（テーブル名プレフィックス等に使用）."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
async def pg_entity_chk(pg_conn: PostgresStoreConnector, test_id: str) -> AsyncIterator[type]:
    """一意なテーブル名を持つチャンク Entity を生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """

    class _TestAppBase(DeclarativeBase):
        metadata = MetaData("app")

    class _TestChunkEntity(_TestAppBase, ChunkFields):
        __tablename__ = f"{test_id}_vectors"

    engine = pg_conn.get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.commit()
    async with engine.begin() as conn:
        await conn.run_sync(_TestAppBase.metadata.create_all)
    yield _TestChunkEntity
    async with engine.begin() as conn:
        await conn.run_sync(_TestAppBase.metadata.drop_all)


@pytest.fixture()
def docs_obs() -> list[Document]:
    """ObsidianReader で docs/dataset_obsidian/ から先頭 10 件を取得するフィクスチャ."""
    return ObsidianLoader(Path("docs/dataset_obsidian/")).load()[:10]


@pytest.fixture()
def llm() -> BaseChatModel:
    """実際の Bedrock LLM に接続した ChatModel.

    AWS_ACCESS_KEY_ID・AWS_SECRET_ACCESS_KEY 環境変数が必要。
    """
    return ChatOpenAI(model="openai.gpt-5.6-luna", reasoning={"effort": "low"})


@pytest.fixture()
async def pg_entity_obs(pg_conn: PostgresStoreConnector, test_id: str) -> AsyncIterator[type]:
    """Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = pg_conn.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestVaultRawEntity(_TestBase, postgres.ObsidianFields):
        __tablename__ = f"{test_id}_raw"

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

    AA_PG_CONNECTION_STRING と AA_GEMINI_API_KEY 環境変数が必要。
    """
    api_key = os.environ.get("AA_GEMINI_API_KEY")
    if not api_key:
        pytest.fail("AA_GEMINI_API_KEY が未設定のため失敗")

    return VaultObsidianRetriever(
        store_conn=pg_conn,
        chunk_entity=pg_entity_chk,
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(api_key),
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
async def pg_entity_smpl(pg_conn: PostgresStoreConnector, test_id: str) -> AsyncIterator[type]:
    """サンプル用 Vault テーブルの ORM エンティティクラスを生成しテーブルを作成する.

    テスト終了後に作成したテーブルを DROP する。
    """
    engine = pg_conn.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestSampleEntity(_TestBase, postgres.DocumentFields):
        __tablename__ = f"{test_id}_raw"

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
    api_key = os.environ.get("AA_GEMINI_API_KEY")
    if not api_key:
        pytest.fail("AA_GEMINI_API_KEY が未設定のため失敗")

    return VaultSampleRetriever(
        chunk_entity=pg_entity_chk,
        store_conn=pg_conn,
        splitter=RecursiveCharacterTextSplitter(),
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(api_key),
        ),
        vault_entity=pg_entity_smpl,
    )
