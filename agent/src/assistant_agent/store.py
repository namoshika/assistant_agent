from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from assistant_agent.utils.absclass import StoreConnector


class PostgresStoreConnector(StoreConnector):
    """PostgreSQL バックエンドの StoreConnector 実装."""

    def __init__(self, connection_string: str) -> None:
        """Construct PostgresStoreConnector."""
        self._url = make_url(connection_string)
        self._engine: AsyncEngine | None = None

    def get_engine(self) -> AsyncEngine:
        """PostgreSQL の SQLAlchemy AsyncEngine を生成する（キャッシュあり）."""
        if self._engine is None:
            self._engine = create_async_engine(self._url.set(drivername="postgresql+asyncpg"))
        return self._engine
