from pathlib import Path

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.storage.docstore.duckdb import DuckDBDocumentStore
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from sqlalchemy import text

from assistant_agent.utils.store_factory import DuckDBStoreContext


class TestDuckDBStoreContext:
    @pytest.mark.integration
    def test_get_engine_01(self, tmp_path: Path) -> None:
        """get_engine が Engine を返すこと.

        観点1: インメモリで Engine が返ること
        観点2: ファイル永続化で Engine が返ること
        観点3: 同一インスタンスが返ること（キャッシュ）
        観点4: 返った Engine で SELECT 1 が実行でき DuckDB と通信できること
        """
        # 観点1: インメモリ
        ctx_mem = DuckDBStoreContext()
        engine_mem = ctx_mem.get_engine()
        assert engine_mem is not None

        # 観点3: キャッシュ（インメモリ）
        assert ctx_mem.get_engine() is engine_mem

        # 観点4: SELECT 1 で通信確認（インメモリ）
        with engine_mem.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1

        # 観点2: ファイル永続化
        ctx_file = DuckDBStoreContext(persist_dir=tmp_path)
        engine_file = ctx_file.get_engine()
        assert engine_file is not None

        # 観点3: キャッシュ（ファイル）
        assert ctx_file.get_engine() is engine_file

        # 観点4: SELECT 1 で通信確認（ファイル）
        with engine_file.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1

        ctx_mem.close()
        ctx_file.close()

    @pytest.mark.integration
    def test_get_vector_store_01(self, tmp_path: Path) -> None:
        """get_vector_store が DuckDBVectorStore を返し、読み書きできること.

        観点1: DuckDBVectorStore インスタンスが返ること
        観点2: add() + query() で実際に DuckDB と読み書きできること
        """
        ctx = DuckDBStoreContext(persist_dir=tmp_path)

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
    def test_get_docstore_01(self, tmp_path: Path) -> None:
        """get_docstore が DuckDBDocumentStore を返し、読み書きできること.

        観点1: DuckDBDocumentStore インスタンスが返ること
        観点2: add_documents() + get_document() で実際に DuckDB と読み書きできること
        """
        ctx = DuckDBStoreContext(persist_dir=tmp_path)

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
    def test_close_01(self, tmp_path: Path) -> None:
        """close() が CHECKPOINT を実行すること.

        観点1: CHECKPOINT により WAL ファイルが消えること
        """
        ctx = DuckDBStoreContext(persist_dir=tmp_path)
        engine = ctx.get_engine()
        wal_entity = tmp_path / "entity.duckdb.wal"
        wal_llama = tmp_path / "llamaindex.duckdb.wal"

        # entity.duckdb に WAL を発生させる
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE _wal_check (id INTEGER)"))
            conn.execute(text("INSERT INTO _wal_check VALUES (1)"))
            conn.commit()
        assert wal_entity.exists()

        # llamaindex.duckdb に WAL を発生させる
        vs = ctx.get_vector_store("vectors", embed_dim=2)
        node = TextNode(id_="n1", text="hello", embedding=[0.1, 0.9])
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc1")
        vs.add([node])
        assert wal_llama.exists()

        # 試験実施
        ctx.close()

        # 結果検証
        # 観点1: CHECKPOINT により両 WAL ファイルが消えること
        assert not wal_entity.exists()
        assert not wal_llama.exists()
