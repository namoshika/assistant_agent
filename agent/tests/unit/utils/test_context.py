import pytest

from assistant_agent.utils.context import ContextRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """テスト間で ContextRegistry._factories をリセットするフィクスチャ."""
    original = {k: dict(v) for k, v in ContextRegistry._factories.items()}
    ContextRegistry._factories.clear()
    yield
    ContextRegistry._factories.clear()
    ContextRegistry._factories.update(original)


def test_register_01():
    """ContextRegistry.register に対しテストすること.

    観点1: @ContextRegistry.register(name) で factory が登録されること
    観点2: variant 省略時は "default" として登録されること
    """

    # 試験準備・実施
    @ContextRegistry.register("my_key")
    def _factory():
        return "value"

    # 結果検証
    # 観点1
    assert "my_key" in ContextRegistry._factories
    # 観点2
    assert "default" in ContextRegistry._factories["my_key"]
    assert ContextRegistry._factories["my_key"]["default"] is _factory


def test_register_02():
    """ContextRegistry.register に対しテストすること（異常系）.

    観点1: 同一 (name, variant) への重複登録で ValueError が送出されること
    観点2: エラーメッセージに name・variant が含まれること
    """

    # 試験準備
    @ContextRegistry.register("my_key")
    def _factory():
        return "value"

    # 試験実施・結果検証
    with pytest.raises(ValueError) as exc_info:

        @ContextRegistry.register("my_key")
        def _factory_dup():
            return "dup"

    msg = str(exc_info.value)
    # 観点1・2
    assert "my_key" in msg
    assert "default" in msg


def test_build_01():
    """ContextRegistry.build に対しテストすること（正常系）.

    観点1: build() で登録済み factory が呼ばれ dict が返ること
    観点2: variants 指定で対応 variant の factory が選択されること
    観点3: 未指定フィールドは "default" を使用すること
    観点4: **kwargs が factory にそのまま渡されること
    """
    # 試験準備
    received_kwargs: dict = {}

    @ContextRegistry.register("key_a")
    def _factory_a_default(**kwargs):
        received_kwargs.update(kwargs)
        return "a_default"

    @ContextRegistry.register("key_a", variant="v1")
    def _factory_a_v1(**kwargs):
        return "a_v1"

    @ContextRegistry.register("key_b")
    def _factory_b_default(**kwargs):
        return "b_default"

    # 試験実施
    result_default = ContextRegistry.build(x=1, y=2)
    result_variant = ContextRegistry.build(variants={"key_a": "v1"})

    # 結果検証
    # 観点1
    assert isinstance(result_default, dict)
    assert result_default["key_a"] == "a_default"  # pyright: ignore[reportGeneralTypeIssues]
    # 観点2
    assert result_variant["key_a"] == "a_v1"  # pyright: ignore[reportGeneralTypeIssues]
    # 観点3
    assert result_variant["key_b"] == "b_default"  # pyright: ignore[reportGeneralTypeIssues]
    # 観点4
    assert received_kwargs == {"x": 1, "y": 2}


def test_build_02():
    """ContextRegistry.build に対しテストすること（異常系）.

    観点1: 存在しない variant を指定した場合に ValueError が送出されること
    観点2: エラーメッセージに variant 名・name・利用可能な variant 一覧が含まれること
    """

    # 試験準備
    @ContextRegistry.register("key_a")
    def _factory():
        return "value"

    # 試験実施・結果検証
    with pytest.raises(ValueError) as exc_info:
        ContextRegistry.build(variants={"key_a": "nonexistent"})

    msg = str(exc_info.value)
    # 観点1・2
    assert "nonexistent" in msg
    assert "key_a" in msg
    assert "default" in msg
