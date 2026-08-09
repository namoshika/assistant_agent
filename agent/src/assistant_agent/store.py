from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from assistant_agent.utils.absclass import StoreConnector


class PostgresStoreConnector(StoreConnector):
    """PostgreSQL バックエンドの StoreConnector 実装."""

    def __init__(self, connection_string: str) -> None:
        """Construct PostgresStoreConnector."""
        self._url = make_url(connection_string)
        self._engine: AsyncEngine | None = None
        self._engine_sync: Engine | None = None

    def get_engine(self) -> AsyncEngine:
        """PostgreSQL の SQLAlchemy AsyncEngine を生成する（キャッシュあり）."""
        if self._engine is None:
            self._engine = create_async_engine(self._url.set(drivername="postgresql+asyncpg"))
        return self._engine

    def get_engine_sync(self) -> Engine:
        """PostgreSQL の SQLAlchemy Engine（同期）を生成する（キャッシュあり）.

        pandas.read_sql 等、同期エンジンを要する用途向け（例: notebook でのレコード表示）。
        """
        if self._engine_sync is None:
            self._engine_sync = create_engine(self._url.set(drivername="postgresql+psycopg"))
        return self._engine_sync

    def get_psycopg_pool(self) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
        """LangGraph checkpointer/store 用の psycopg AsyncConnectionPool を返す（未オープン）."""
        psycopg_conn_string = self._url.set(drivername="postgresql")
        psycopg_conn_string = psycopg_conn_string.render_as_string(hide_password=False)
        return AsyncConnectionPool(
            psycopg_conn_string,
            connection_class=AsyncConnection[DictRow],
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
                # checkpoints 系テーブルを public スキーマへ作成させないため、
                # 接続の search_path を PSYCOPG_SCHEMA へ固定する。
                "options": "-c search_path=app",
            },
            # チェックアウト時に生存確認 (check) を行い、スリープ復帰後などで
            # サーバー側から切断済みのコネクションが再利用されるのを防ぐ。
            check=self._check_connection,
            open=False,
        )

    @staticmethod
    async def _check_connection(conn: AsyncConnection[DictRow]) -> None:
        """接続が生きているかを確認する (psycopg_pool.check_connection 相当)."""
        # ライブラリ側の型定義が AsyncConnection[TupleRow] 固定になっており
        # AsyncConnection[DictRow] を渡すと Pyright エラーになるため、
        # 実装 (SELECT 1 相当の execute) をここに複製する。
        await conn.execute("")
