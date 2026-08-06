from unittest.mock import MagicMock

from assistant_agent.store import PostgresStoreConnector
from assistant_agent.utils.context import ContextRegistry


def test_context_registry_includes_services_01() -> None:
    """agent_bot import 済み状態で ContextRegistry.build() が全 factory を含む context を返すこと.

    登録発火 import の移設漏れがあると build() が空 dict を返す（例外にはならない）ため、
    実際にキーが存在することを確認する。
    """
    # 試験準備
    store_conn = MagicMock(spec=PostgresStoreConnector)

    # 試験実施
    import assistant_agent.agent_bot  # noqa: F401  登録発火 import の実行を保証するため

    ctx = ContextRegistry.build(store_conn=store_conn)

    # 結果検証
    assert "dispatcher_service" in ctx
    assert "discord_service" in ctx
