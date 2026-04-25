from llama_index.core.storage.docstore.types import BaseDocumentStore
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

    LlamaIndex と SQLAlchemy Engine は同一ファイルへの同時接続が不可のため
    ファイルを分けて管理する:
        LlamaIndex 用: {persist_dir}/llamaindex.duckdb（VectorStore / Docstore 共有接続）
        SQLAlchemy 用: {persist_dir}/entity.duckdb

    永続化モード毎に、各オブジェクトを以下の様に初期化する.
    persist_dir: ":memory:" (デフォルト)
        create_engine(url="duckdb:///:memory:")
        DuckDBVectorStore(database_name=":memory:")           # persist_dir 引数は渡さない
        DuckDBKVStore(database_name=":memory:")               # persist_dir 引数は渡さない
    persist_dir: "(path)"
        create_engine(url="duckdb:///(path)/entity.duckdb")
        DuckDBVectorStore(database_name="llamaindex.duckdb", persist_dir="(path)")
        DuckDBKVStore(database_name="llamaindex.duckdb", persist_dir="(path)")
    """

    def __init__(self, persist_dir: str = ":memory:", read_only: bool = False) -> None:
        """Construct DuckDBStoreContext."""
        self._persist_dir = persist_dir
        is_memory = persist_dir == ":memory:"
        self._is_memory = is_memory
        self._dbname = ":memory:" if is_memory else "llamaindex.duckdb"
        effective_read_only = False if is_memory else read_only
        self._engine: Engine = create_engine(
            url="duckdb:///:memory:" if is_memory else f"duckdb:///{persist_dir}/entity.duckdb",
            connect_args={"read_only": effective_read_only},
        )

    def get_engine(self) -> Engine:
        """duckdb-engine 経由の SQLAlchemy Engine を返す."""
        return self._engine

    def get_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
        """DuckDBVectorStore を生成する."""
        if self._is_memory:
            return DuckDBVectorStore(self._dbname, name, embed_dim)
        else:
            return DuckDBVectorStore(
                self._dbname, name, embed_dim, persist_dir=str(self._persist_dir)
            )

    def get_docstore(self, name: str) -> BaseDocumentStore:
        """DuckDBDocumentStore を生成する."""
        if self._is_memory:
            kvstore = DuckDBKVStore(self._dbname, name)
        else:
            kvstore = DuckDBKVStore(self._dbname, name, persist_dir=str(self._persist_dir))
        return DuckDBDocumentStore(duckdb_kvstore=kvstore)

    def flush(self) -> None:
        """CHECKPOINT で WAL をフラッシュする.

        CHECKPOINT はデータ永続化の代替ではなく安全策。
        呼び出し元が session.commit() を適切に呼ぶことが前提。
        """
        with self._engine.connect() as conn:
            conn.execute(text("CHECKPOINT"))

    def close(self) -> None:
        """Engine の接続を解放する."""
        self.flush()
        self._engine.dispose()
