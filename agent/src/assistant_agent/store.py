from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine.url import make_url

from assistant_agent.utils.absclass import StoreConnector


class PostgresStoreConnector(StoreConnector):
    """PostgreSQL バックエンドの StoreConnector 実装."""

    def __init__(self, connection_string: str) -> None:
        """Construct PostgresStoreConnector."""
        self._url = make_url(connection_string)
        self._engine: Engine | None = None
        self._pg_engine: PGEngine | None = None

    def get_engine(self) -> Engine:
        """PostgreSQL の SQLAlchemy Engine を生成する（キャッシュあり）."""
        if self._engine is None:
            self._engine = create_engine(self._url)
        return self._engine

    def get_vector_store(self, entity: type, embedding: Embeddings) -> VectorStore:
        """PGVectorStore を生成する.

        PGEngine.from_engine() は AsyncEngine 専用であり、get_engine() が返す同期 Engine を
        ラップできないため、PGEngine は接続文字列から別途生成する（実機確認済み）。
        """
        if self._pg_engine is None:
            self._pg_engine = PGEngine.from_connection_string(self._url)
        return PGVectorStore.create_sync(
            engine=self._pg_engine,
            embedding_service=embedding,
            table_name=entity.__tablename__,
            schema_name=entity.metadata.schema,
            # 空リストは falsy 判定されるため、metadata_columns 自動認識を確実に発動させる目的で
            # metadata_json_column（JSON 列としての扱いは変わらない）を明示的に指定する
            ignore_metadata_columns=["langchain_metadata"],
        )
