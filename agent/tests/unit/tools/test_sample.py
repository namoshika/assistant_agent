from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pytest_mock import MockerFixture

import assistant_agent.tools.sample as tools


class _FakeChatModel(GenericFakeChatModel):
    """bind_tools() をサポートするダミーチャットモデル."""

    def bind_tools(self, tools, **_):
        return self


def test_get_weather_01():
    """天気を取得できるか確認.

    観点: ToolMessage として返り、content に引数の都市名が含まれる
    """
    # 試験準備
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"city": "Tokyo"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.get_weather)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="東京の天気は?")]})
    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Tokyo" in tool_msg.content


def test_get_datetime_now_01(mocker: MockerFixture):
    """現在時刻を JST の ISO8601 文字列として取得できるか確認.

    観点: ToolMessage の content が JST の ISO8601 形式の現在時刻と一致する
    """
    # 試験準備
    fixed_now = datetime(2026, 8, 1, 21, 34, 56, tzinfo=ZoneInfo("Asia/Tokyo"))
    mocker.patch("assistant_agent.tools.sample.datetime").now.return_value = fixed_now
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_datetime_now",
                "args": {},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.get_datetime_now)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="今何時?")]})

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "2026-08-01T21:34:56+09:00"


def _make_agent(tool_calls_msg: AIMessage, *tools, context_schema=None):
    """テスト用エージェントを生成するヘルパー."""
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools), context_schema=context_schema)
