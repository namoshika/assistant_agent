import os

import pytest
from sqlalchemy import text

from assistant_agent.store import PostgresStoreConnector


class TestPostgresStoreConnector:
    @pytest.mark.integration
    async def test_get_engine_01(self) -> None:
        """get_engine が AsyncEngine を返すこと.

        観点1: AsyncEngine インスタンスが返ること
        観点2: 同一インスタンスが返ること（キャッシュ）
        観点3: SELECT 1 で PostgreSQL と通信できること
        """
        conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
        if not conn_str:
            pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
        ctx = PostgresStoreConnector(conn_str)

        # 試験実施
        engine = ctx.get_engine()

        # 結果検証
        # 観点1
        assert engine is not None
        # 観点2
        assert ctx.get_engine() is engine
        # 観点3
        async with engine.connect() as conn:
            result = (await conn.execute(text("SELECT 1"))).scalar()
        assert result == 1
        await engine.dispose()
