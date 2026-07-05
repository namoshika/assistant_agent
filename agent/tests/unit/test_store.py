from unittest.mock import MagicMock, patch

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from assistant_agent.entities.postgres import ChunkFields
from assistant_agent.store import PostgresStoreConnector


class _Base(DeclarativeBase):
    metadata = MetaData("app")


class _DummyChunkEntity(_Base, ChunkFields):
    __tablename__ = "vec_store"


class TestPostgresStoreConnector:
    def test_get_engine_01(self) -> None:
        """get_engine が接続文字列から Engine を生成し、キャッシュすること.

        観点1: create_engine が接続文字列 URL で呼ばれること
        観点2: 1回目と2回目の呼び出しが同一インスタンスを返すこと
        """
        mock_engine = MagicMock()
        with patch("assistant_agent.store.create_engine", return_value=mock_engine) as mock_ce:
            ctx = PostgresStoreConnector("postgresql://u:p@host:5432/db")

            # 試験実施
            engine1 = ctx.get_engine()
            engine2 = ctx.get_engine()

            # 結果検証
            # 観点1
            assert mock_ce.call_args.args[0].database == "db"
            # 観点2
            assert engine1 is engine2
            mock_ce.assert_called_once()

    def test_get_vector_store_01(self, mocker) -> None:
        """create_sync で VectorStore を取得できること.

        観点1: PGVectorStore.create_sync が entity の __tablename__/schema で呼ばれること
        """
        mock_pg_engine = MagicMock()
        mock_vs = MagicMock()
        mocker.patch("assistant_agent.store.create_engine")
        mocker.patch(
            "assistant_agent.store.PGEngine.from_connection_string", return_value=mock_pg_engine
        )
        mock_create_sync = mocker.patch(
            "assistant_agent.store.PGVectorStore.create_sync", return_value=mock_vs
        )
        mock_embedding = MagicMock()

        ctx = PostgresStoreConnector("postgresql://u:p@host:5432/db")

        # 試験実施
        result = ctx.get_vector_store(_DummyChunkEntity, mock_embedding)

        # 結果検証
        # 観点1
        mock_create_sync.assert_called_once_with(
            engine=mock_pg_engine,
            embedding_service=mock_embedding,
            table_name="vec_store",
            schema_name="app",
            ignore_metadata_columns=["langchain_metadata"],
        )
        assert result is mock_vs
