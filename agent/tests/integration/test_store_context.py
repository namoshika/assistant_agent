import os

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from assistant_agent.store import PostgresStoreConnector


class TestPostgresStoreConnector:
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
        ctx = PostgresStoreConnector(conn_str)

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
    def test_get_vector_store_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_chk: type
    ) -> None:
        """get_vector_store が PGVectorStore を返し、読み書きできること.

        観点1: add_documents() + similarity_search() で実際に PostgreSQL と読み書きできること
        """
        embed_dim = pg_entity_chk.__table__.c.embedding.type.dim

        # 試験実施
        store = pg_conn.get_vector_store(pg_entity_chk, DeterministicFakeEmbedding(size=embed_dim))
        store.add_documents(
            [
                Document(
                    page_content="hello",
                    metadata={"document_id": "doc-1", "document_content_hash": "hash-1", "k": "v"},
                )
            ]
        )

        # 結果検証
        # 観点1
        result = store.similarity_search("hello", k=1)
        assert len(result) == 1
        assert result[0].page_content == "hello"
