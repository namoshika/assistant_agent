from assistant_agent.utils.store_context import DuckDBStoreContext


def test_get_engine_01() -> None:
    """get_engine がインメモリ Engine を返しキャッシュすること.

    観点1: Engine インスタンスが返ること
    観点2: 同一インスタンスが返ること（キャッシュ）
    """
    ctx = DuckDBStoreContext()

    # 試験実施
    engine1 = ctx.get_engine()
    engine2 = ctx.get_engine()

    # 結果検証
    # 観点1
    assert engine1 is not None
    # 観点2
    assert engine1 is engine2

    ctx.close()
