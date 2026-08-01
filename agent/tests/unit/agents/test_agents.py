import os

from langgraph.graph.state import CompiledStateGraph
from pytest_mock import MockerFixture

from assistant_agent import agents


def test_build_agent_01(mocker: MockerFixture):
    """build_agent() が lc_agent・context のペアを返せるか確認.

    観点1: 戻り値の1つ目が CompiledStateGraph インスタンスであること
    観点2: 戻り値の2つ目（context）が dict であること
    """
    # 試験準備
    mocker.patch.dict(
        os.environ,
        {
            "AA_GEMINI_API_KEY": "dummy-key",
            "AA_PG_CONNECTION_STRING": "postgresql://localhost/test",
            "AA_DISCORD_BOT_TOKEN": "dummy-token",
        },
    )

    # 試験実施
    lc_agent, ctx = agents.build_agent()

    # 結果検証
    # 観点1
    assert isinstance(lc_agent, CompiledStateGraph)
    # 観点2
    assert isinstance(ctx, dict)
