from unittest.mock import MagicMock, patch

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
