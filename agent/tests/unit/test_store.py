from unittest.mock import MagicMock, patch

from assistant_agent.store import DuckDBStoreConnector


class TestDuckDBStoreConnector:
    def test_persist_01(self) -> None:
        """DuckDBStoreConnector がインメモリモードで各コンポーネントを初期化すること.

        観点1: get_engine が create_engine(url="duckdb:///:memory:") で初期化すること
        観点2: get_engine の1回目と2回目が同一インスタンスを返すこと
        観点3: get_vector_store が DuckDBVectorStore(":memory:", name, embed_dim) で初期化すること
               （persist_dir 引数なし）
        観点4: get_docstore が DuckDBKVStore(":memory:", name) で初期化すること
               （persist_dir 引数なし）
        """
        mock_engine = MagicMock()
        mock_vs = MagicMock()
        mock_kvstore = MagicMock()
        mock_docstore = MagicMock()

        with (
            patch(
                "assistant_agent.store.create_engine", return_value=mock_engine
            ) as mock_ce,
            patch(
                "assistant_agent.store.DuckDBVectorStore", return_value=mock_vs
            ) as mock_dv,
            patch(
                "assistant_agent.store.DuckDBKVStore", return_value=mock_kvstore
            ) as mock_kv,
            patch(
                "assistant_agent.store.DuckDBDocumentStore",
                return_value=mock_docstore,
            ),
        ):
            ctx = DuckDBStoreConnector()

            # 観点1
            mock_ce.assert_called_once_with(
                url="duckdb:///:memory:", connect_args={"read_only": False}
            )

            # 観点2
            engine1 = ctx.get_engine()
            engine2 = ctx.get_engine()
            assert engine1 is engine2

            # 観点3
            ctx.get_vector_store("vec_store", embed_dim=4)
            mock_dv.assert_called_once_with(":memory:", "vec_store", 4)

            # 観点4
            ctx.get_docstore("doc_store")
            mock_kv.assert_called_once_with(":memory:", "doc_store")

    def test_persist_02(self, tmp_path) -> None:
        """DuckDBStoreConnector がファイル永続化モードで各コンポーネントを初期化すること.

        観点1: get_engine が create_engine(url="duckdb:///{path}/entity.duckdb") で初期化すること
        観点2: get_engine の1回目と2回目が同一インスタンスを返すこと
        観点3: get_vector_store が DuckDBVectorStore("llamaindex.duckdb", name, embed_dim,
               persist_dir="{path}") で初期化すること
        観点4: get_docstore が DuckDBKVStore("llamaindex.duckdb", name, persist_dir="{path}")
               で初期化すること
        """
        persist_dir = str(tmp_path)
        mock_engine = MagicMock()
        mock_vs = MagicMock()
        mock_kvstore = MagicMock()
        mock_docstore = MagicMock()

        with (
            patch(
                "assistant_agent.store.create_engine", return_value=mock_engine
            ) as mock_ce,
            patch(
                "assistant_agent.store.DuckDBVectorStore", return_value=mock_vs
            ) as mock_dv,
            patch(
                "assistant_agent.store.DuckDBKVStore", return_value=mock_kvstore
            ) as mock_kv,
            patch(
                "assistant_agent.store.DuckDBDocumentStore",
                return_value=mock_docstore,
            ),
        ):
            ctx = DuckDBStoreConnector(persist_dir=persist_dir)

            # 観点1
            mock_ce.assert_called_once_with(
                url=f"duckdb:///{persist_dir}/entity.duckdb",
                connect_args={"read_only": False},
            )

            # 観点2
            engine1 = ctx.get_engine()
            engine2 = ctx.get_engine()
            assert engine1 is engine2

            # 観点3
            ctx.get_vector_store("vec_store", embed_dim=4)
            mock_dv.assert_called_once_with(
                "llamaindex.duckdb", "vec_store", 4, persist_dir=persist_dir
            )

            # 観点4
            ctx.get_docstore("doc_store")
            mock_kv.assert_called_once_with(
                "llamaindex.duckdb", "doc_store", persist_dir=persist_dir
            )
