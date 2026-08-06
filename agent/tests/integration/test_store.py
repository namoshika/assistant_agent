import uuid
from collections.abc import AsyncIterator

import pytest
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_agent.entities.checkpoint import CheckpointEntity
from assistant_agent.store import PostgresStoreConnector


@pytest.fixture
def thread_id() -> str:
    """テストごとに一意な thread_id."""
    return f"test-checkpoint-entity:{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def cleanup(pg_conn: PostgresStoreConnector) -> AsyncIterator[None]:
    """テスト終了後、AsyncPostgresSaver.setup() が作成したテーブルを DROP する."""
    yield
    engine = pg_conn.get_engine()
    async with engine.begin() as conn:
        for table_name in (
            "checkpoint_writes",
            "checkpoint_blobs",
            "checkpoints",
            "checkpoint_migrations",
        ):
            await conn.execute(text(f"DROP TABLE IF EXISTS app.{table_name}"))


class TestPostgresStoreConnector:
    @pytest.mark.integration
    async def test_get_engine_01(self, pg_conn: PostgresStoreConnector) -> None:
        """get_engine が AsyncEngine を返すこと.

        観点1: AsyncEngine インスタンスが返ること
        観点2: 同一インスタンスが返ること（キャッシュ）
        観点3: SELECT 1 で PostgreSQL と通信できること
        """
        # 試験実施
        engine = pg_conn.get_engine()

        # 結果検証
        # 観点1
        assert engine is not None
        # 観点2
        assert pg_conn.get_engine() is engine
        # 観点3
        async with engine.connect() as conn:
            result = (await conn.execute(text("SELECT 1"))).scalar()
        assert result == 1

    @pytest.mark.integration
    async def test_get_psycopg_pool_01(self, pg_conn: PostgresStoreConnector) -> None:
        """get_psycopg_pool が実際に PostgreSQL へオープンできること.

        観点1: SELECT 1 で PostgreSQL と通信できること
        """
        # 試験実施
        async with pg_conn.get_psycopg_pool() as pool, pool.connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()

        # 結果検証
        # 観点1
        assert result == {"?column?": 1}

    @pytest.mark.integration
    async def test_get_psycopg_pool_02(
        self, pg_conn: PostgresStoreConnector, thread_id: str
    ) -> None:
        """get_psycopg_pool の接続で AsyncPostgresSaver が書き込んだ checkpoint を CheckpointEntity 経由の SELECT で読み出せること.

        観点1: 書き込んだ thread_id・checkpoint_id が SELECT 結果に含まれること
        """  # noqa: E501
        checkpoint: Checkpoint = empty_checkpoint()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        # 試験実施
        async with pg_conn.get_psycopg_pool() as pool:
            saver = AsyncPostgresSaver(conn=pool)
            await saver.setup()
            await saver.aput(config, checkpoint, {}, {})

        # 結果検証
        # 観点1
        async with AsyncSession(pg_conn.get_engine()) as session:
            rows = (
                await session.scalars(
                    select(CheckpointEntity).where(CheckpointEntity.thread_id == thread_id)
                )
            ).all()
        assert len(rows) == 1
        assert rows[0].thread_id == thread_id
        assert rows[0].checkpoint_id == checkpoint["id"]
