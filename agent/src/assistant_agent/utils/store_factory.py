from pathlib import Path

import duckdb
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.storage.docstore.duckdb import DuckDBDocumentStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine.url import make_url

from assistant_agent.utils.absclass import StoreContext


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

    def get_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
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

    def get_docstore(self, name: str) -> BaseDocumentStore:
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


class DuckDBStoreContext(StoreContext):
    """DuckDB バックエンドの StoreContext 実装.

    persist_dir=None でインメモリ、Path 指定でファイル永続化。
    LlamaIndex と SQLAlchemy Engine は同一ファイルへの同時接続が不可のため
    ファイルを分けて管理する:
        LlamaIndex 用: {persist_dir}/llamaindex.duckdb（VectorStore / Docstore 共有接続）
        SQLAlchemy 用: {persist_dir}/entity.duckdb

    利用終了時:
        factory.close()  # CHECKPOINT を実行し WAL を安全にフラッシュ
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Construct DuckDBStoreContext."""
        self._persist_dir = persist_dir
        self._dbname = ":memory:" if persist_dir is None else "llamaindex.duckdb"
        self._conn = duckdb.connect(
            ":memory:" if persist_dir is None else str(persist_dir / self._dbname)
        )
        self._engine: Engine = create_engine(
            "duckdb:///:memory:"
            if persist_dir is None
            else f"duckdb:///{persist_dir / 'entity.duckdb'}"
        )

    def get_engine(self) -> Engine:
        """duckdb-engine 経由の SQLAlchemy Engine を返す."""
        return self._engine

    def get_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
        """DuckDBVectorStore を生成する."""
        return DuckDBVectorStore(
            self._dbname, name, embed_dim, persist_dir=str(self._persist_dir), client=self._conn
        )

    def get_docstore(self, name: str) -> BaseDocumentStore:
        """DuckDBDocumentStore を生成する."""
        kvstore = DuckDBKVStore(
            self._dbname, name, persist_dir=str(self._persist_dir), client=self._conn
        )
        return DuckDBDocumentStore(duckdb_kvstore=kvstore)

    def close(self) -> None:
        """Engine の接続を解放し CHECKPOINT で WAL をフラッシュする.

        CHECKPOINT はデータ永続化の代替ではなく安全策。
        呼び出し元が session.commit() を適切に呼ぶことが前提。
        """
        self._conn.execute("CHECKPOINT")
        self._conn.close()
        with self._engine.connect() as conn:
            conn.execute(text("CHECKPOINT"))
        self._engine.dispose()


class InMemoryStoreContext(StoreContext):
    """オンメモリの StoreContext 実装（テスト・Databricks 暫定用途）.

    ステートレス設計: 呼び出し毎に新規インスタンスを生成し、単一の Factory から
    異なるパラメータを持つ複数の Retriever を作成できる。

    persist_dir を指定すると、get_vector_store / get_docstore 呼び出し時に
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

    def get_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
        """SimpleVectorStore を生成する（ファイルが存在する場合は自動ロード）."""
        return SimpleVectorStore()

    def get_docstore(self, name: str) -> BaseDocumentStore:
        """SimpleDocumentStore を生成する（ファイルが存在する場合は自動ロード）."""
        return SimpleDocumentStore()
