from unittest.mock import MagicMock, patch

from pytest_mock import MockerFixture

from assistant_agent.store import PostgresStoreConnector


class TestPostgresStoreConnector:
    def test_get_engine_01(self) -> None:
        """get_engine が接続文字列から AsyncEngine を生成し、キャッシュすること.

        観点1: create_async_engine が postgresql+asyncpg ドライバの URL で呼ばれること
        観点2: 1回目と2回目の呼び出しが同一インスタンスを返すこと
        """
        mock_engine = MagicMock()
        with patch(
            "assistant_agent.store.create_async_engine", return_value=mock_engine
        ) as mock_cae:
            ctx = PostgresStoreConnector("postgresql://u:p@host:5432/db")

            # 試験実施
            engine1 = ctx.get_engine()
            engine2 = ctx.get_engine()

            # 結果検証
            # 観点1
            assert mock_cae.call_args.args[0].database == "db"
            assert mock_cae.call_args.args[0].drivername == "postgresql+asyncpg"
            # 観点2
            assert engine1 is engine2
            mock_cae.assert_called_once()

    def test_get_psycopg_pool_01(self, mocker: MockerFixture) -> None:
        """get_psycopg_pool が open=False の AsyncConnectionPool を構築すること.

        観点1: 接続文字列が psycopg 互換の postgresql:// スキームへ変換されて渡ること
            （postgresql+psycopg:// 等の SQLAlchemy 用ドライバ指定は psycopg が解釈できないため）
        観点2: public スキーマを使わせないため search_path=app を options に指定すること
        観点3: チェックアウト時の生存確認として check コールバックを指定すること
        """
        mock_pool_cls = mocker.patch("assistant_agent.store.AsyncConnectionPool")
        ctx = PostgresStoreConnector("postgresql+psycopg://u:p@host:5432/db")

        # 試験実施
        pool = ctx.get_psycopg_pool()

        # 結果検証
        # 観点1、観点2、観点3
        mock_pool_cls.assert_called_once_with(
            "postgresql://u:p@host:5432/db",
            connection_class=mocker.ANY,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": mocker.ANY,
                "options": "-c search_path=app",
            },
            check=mocker.ANY,
            open=False,
        )
        assert pool is mock_pool_cls.return_value
