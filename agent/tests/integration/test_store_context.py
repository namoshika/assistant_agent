import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.storage.docstore.duckdb import DuckDBDocumentStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import text

from assistant_agent.utils.store_context import DuckDBStoreContext, PostgresStoreContext


class TestPostgresStoreContext:
    @pytest.mark.integration
    def test_get_engine_01(self) -> None:
        """get_engine が Engine を返すこと.

        観点1: Engine インスタンスが返ること
        観点2: 同一インスタンスが返ること（キャッシュ）
        観点3: SELECT 1 で PostgreSQL と通信できること
        """
        conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
        if not conn_str:
            pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
        ctx = PostgresStoreContext(conn_str, schema_name="app")

        # 試験実施
        engine = ctx.get_engine()

        # 結果検証
        # 観点1
        assert engine is not None
        # 観点2
        assert ctx.get_engine() is engine
        # 観点3
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1
        engine.dispose()

    @pytest.mark.integration
    def test_get_vector_store_01(self) -> None:
        """get_vector_store が PGVectorStore を返し、読み書きできること.

        観点1: PGVectorStore インスタンスが返ること
        観点2: add() + query() で実際に PostgreSQL と読み書きできること
        """
        conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
        if not conn_str:
            pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
        name = f"test_{uuid.uuid4().hex[:8]}_vec"
        ctx = PostgresStoreContext(conn_str, schema_name="app")
        engine = ctx.get_engine()

        # 試験実施
        store = ctx.get_vector_store(name, embed_dim=3)

        # 結果検証
        # 観点1
        assert isinstance(store, PGVectorStore)
        # 観点2
        node = TextNode(id_="node-1", text="hello", embedding=[0.1, 0.2, 0.3])
        store.add([node])
        result = store.query(VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=1))
        assert len(result.nodes) == 1  # pyright: ignore[reportArgumentType]
        assert result.nodes[0].node_id == "node-1"  # pyright: ignore[reportOptionalSubscript]
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS app.data_{name} CASCADE"))
            conn.commit()
        engine.dispose()

    @pytest.mark.integration
    def test_get_docstore_01(self) -> None:
        """get_docstore が PostgresDocumentStore を返し、読み書きできること.

        観点1: PostgresDocumentStore インスタンスが返ること
        観点2: add_documents() + get_document() で実際に PostgreSQL と読み書きできること
        """
        conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
        if not conn_str:
            pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
        name = f"test_{uuid.uuid4().hex[:8]}_doc"
        ctx = PostgresStoreContext(conn_str, schema_name="app")
        engine = ctx.get_engine()

        # 試験実施
        store = ctx.get_docstore(name)

        # 結果検証
        # 観点1
        assert isinstance(store, PostgresDocumentStore)
        # 観点2
        node = TextNode(id_="node-2", text="world")
        store.add_documents([node])
        fetched = store.get_document("node-2")
        assert fetched is not None
        assert fetched.get_content() == "world"
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS app.data_{name} CASCADE"))
            conn.commit()
        engine.dispose()


class TestDuckDBStoreContext:
    @pytest.fixture
    def ctx(self, request, tmp_path: Path) -> Iterator[DuckDBStoreContext]:
        """DuckDBStoreContext インスタンスを生成するフィクスチャ.

        request.param が ":memory:" の場合はインメモリモード、
        それ以外の場合は tmp_path を persist_dir として使用する。
        """
        if request.param == ":memory:":
            instance = DuckDBStoreContext(persist_dir=":memory:")
        else:
            instance = DuckDBStoreContext(persist_dir=str(tmp_path))
        yield instance
        instance.close()

    @pytest.mark.integration
    @pytest.mark.parametrize("ctx", [":memory:", "file"], indirect=True)
    def test_get_engine_01(self, ctx: DuckDBStoreContext) -> None:
        """get_engine が Engine を返すこと.

        観点1: Engine インスタンスが返ること
        観点2: 同一インスタンスが返ること（キャッシュ）
        観点3: 返った Engine で SELECT 1 が実行でき DuckDB と通信できること
        """
        # 試験実施
        engine = ctx.get_engine()

        # 結果検証
        # 観点1
        assert engine is not None
        # 観点2
        assert ctx.get_engine() is engine
        # 観点3
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1

    @pytest.mark.integration
    def test_get_engine_02(self, tmp_path: Path) -> None:
        """get_engine が Read Only モードで Engine を返すこと.

        観点1: Read Only で Engine が返り SELECT 1 が通ること
        """
        # 試験準備: Read Only で開く前に DB ファイルを作成しておく
        ctx_w = DuckDBStoreContext(persist_dir=str(tmp_path))
        ctx_w.close()

        ctx = DuckDBStoreContext(persist_dir=str(tmp_path), read_only=True)

        # 試験実施
        engine = ctx.get_engine()

        # 結果検証
        # 観点1
        assert engine is not None
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1
        ctx.close()

    @pytest.mark.integration
    @pytest.mark.parametrize("ctx", [":memory:", "file"], indirect=True)
    def test_get_vector_store_01(self, ctx: DuckDBStoreContext) -> None:
        """get_vector_store が DuckDBVectorStore を返し、読み書きできること.

        観点1: DuckDBVectorStore インスタンスが返ること
        観点2: add() + query() で実際に DuckDB と読み書きできること
        """
        # 試験実施
        store = ctx.get_vector_store("test_vec", embed_dim=3)

        # 結果検証
        # 観点1
        assert isinstance(store, DuckDBVectorStore)
        # 観点2
        node = TextNode(id_="node-1", text="hello", embedding=[0.1, 0.2, 0.3])
        store.add([node])
        result = store.query(VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=1))
        assert len(result.nodes) == 1  # pyright: ignore[reportArgumentType]
        assert result.nodes[0].node_id == "node-1"  # pyright: ignore[reportOptionalSubscript]

    @pytest.mark.integration
    def test_get_vector_store_02(self, tmp_path: Path) -> None:
        """DuckDBVectorStore が client 引数で渡した接続を無視すること（llamaindex バグの再現確認）.

        観点1: add() の書き込みが渡した接続には反映されないこと

        llamaindex のバグ（Pydantic PrivateAttr リセット）により client 引数が無視され
        独自接続が生成される。バグ修正時にこのテストが失敗するようになる。
        """
        # 試験準備: llamaindex.duckdb とは別ファイルを向く接続を用意
        other_db = tmp_path / "other.duckdb"
        conn = duckdb.connect(str(other_db))
        conn.execute(
            "CREATE TABLE test_vec"
            " (node_id VARCHAR PRIMARY KEY, text TEXT, embedding FLOAT[3], metadata_ JSON)"
        )
        vs = DuckDBVectorStore(
            "llamaindex.duckdb", "test_vec", 3, persist_dir=str(tmp_path), client=conn
        )

        # 試験実施
        vs.add([TextNode(id_="node-1", text="hello", embedding=[0.1, 0.2, 0.3])])

        # 結果検証
        # 観点1: バグにより渡した接続（other.duckdb）には書き込まれないこと
        rows = conn.execute("SELECT count(*) FROM test_vec").fetchone()[0]  # pyright: ignore[reportOptionalSubscript]
        assert rows == 0

    @pytest.mark.integration
    @pytest.mark.parametrize("ctx", [":memory:", "file"], indirect=True)
    def test_get_docstore_01(self, ctx: DuckDBStoreContext) -> None:
        """get_docstore が DuckDBDocumentStore を返し、読み書きできること.

        観点1: DuckDBDocumentStore インスタンスが返ること
        観点2: add_documents() + get_document() で実際に DuckDB と読み書きできること
        """
        # 試験実施
        store = ctx.get_docstore("test_doc")

        # 結果検証
        # 観点1
        assert isinstance(store, DuckDBDocumentStore)
        # 観点2
        node = TextNode(id_="node-2", text="world")
        store.add_documents([node])
        fetched = store.get_document("node-2")
        assert fetched is not None
        assert fetched.get_content() == "world"

    @pytest.mark.integration
    @pytest.mark.parametrize("ctx", [":memory:", "file"], indirect=True)
    def test_close_01(self, ctx: DuckDBStoreContext) -> None:
        """close() が正常終了すること.

        観点1: 例外が発生せずに完了すること
        """
        ctx.get_engine()

        # 試験実施・結果検証
        # 観点1: 例外が発生せずに完了すること
        ctx.close()
