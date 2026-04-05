from pathlib import Path

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine.url import make_url

from agent_assistant.utils.absclass import StoreContext


class PostgresStoreContext(StoreContext):
    """PostgreSQL バックエンドの StoreContext 実装."""

    def __init__(self, connection_string: str, schema_name: str) -> None:
        """Construct PostgresStoreContext."""
        self._url = make_url(connection_string)
        self._schema_name = schema_name
        self._engine: Engine | None = None

    def get_engine(self) -> Engine:
        """PostgreSQL の SQLAlchemy Engine を生成する（キャッシュあり）."""
        if self._engine is None:
            self._engine = create_engine(self._url)
        return self._engine

    def create_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
        """PGVectorStore を生成する."""
        return PGVectorStore.from_params(
            host=self._url.host,
            port=str(self._url.port or 5432),
            database=self._url.database,
            user=self._url.username,
            password=str(self._url.password or ""),
            table_name=name,
            embed_dim=embed_dim,
            schema_name=self._schema_name,
            use_jsonb=True,
        )

    def create_docstore(self, name: str) -> BaseDocumentStore:
        """PostgresDocumentStore を生成する."""
        return PostgresDocumentStore.from_params(
            host=self._url.host,
            port=str(self._url.port or 5432),
            database=self._url.database,
            user=self._url.username,
            password=str(self._url.password or ""),
            table_name=name,
            schema_name=self._schema_name,
            use_jsonb=True,
        )


class InMemoryStoreContext(StoreContext):
    """オンメモリの StoreContext 実装（テスト・Databricks 暫定用途）.

    ステートレス設計: 呼び出し毎に新規インスタンスを生成し、単一の Factory から
    異なるパラメータを持つ複数の Retriever を作成できる。

    persist_dir を指定すると、create_vector_store / create_docstore 呼び出し時に
    対応するファイルが存在すれば自動ロードする。
    ストアの永続化が必要な場合は呼び出し元が直接 store.persist(path) を呼ぶこと。
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Construct InMemoryStoreContext."""
        self._persist_dir = persist_dir
        self._engine: Engine | None = None

    def get_engine(self) -> Engine:
        """SQLite インメモリの Engine を生成する（キャッシュあり）.

        sqlite:///:memory: は Engine ごとに独立した DB になるため、
        キャッシュにより同一 Factory を使う呼び出し元が同一 DB を共有できる。
        """
        if self._engine is None:
            self._engine = create_engine("sqlite:///:memory:")
        return self._engine

    def create_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
        """SimpleVectorStore を生成する（ファイルが存在する場合は自動ロード）."""
        return SimpleVectorStore()

    def create_docstore(self, name: str) -> BaseDocumentStore:
        """SimpleDocumentStore を生成する（ファイルが存在する場合は自動ロード）."""
        return SimpleDocumentStore()
