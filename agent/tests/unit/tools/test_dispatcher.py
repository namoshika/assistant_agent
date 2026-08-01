from datetime import UTC, datetime
from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import assistant_agent.tools.dispatcher as tools
from assistant_agent.services.dispatcher import Dispatch, DispatcherService, IntervalUnit


class _FakeChatModel(GenericFakeChatModel):
    """bind_tools() をサポートするダミーチャットモデル."""

    def bind_tools(self, tools, **_):
        return self


async def test_dispatcher_invoke_at_01():
    """ツール dispatcher_invoke_at が DispatcherService.invoke_at() を正しい引数で呼ぶことを確認.

    観点1（R017）: content・datetime を含む形で invoke_at() が呼ばれること
    観点2（R017）: invoke_at() の戻り値（Dispatch）の文字列表現が LLM への ToolMessage に
      含まれること
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    at = datetime(2026, 7, 30, 9, 0, 0, tzinfo=UTC)
    m_service.invoke_at.return_value = Dispatch(
        dispatch_id="dispatch-1",
        invocation={"input": {"messages": [HumanMessage(content="おはよう")]}},
        interval_seconds=DispatcherService.ONE_SHOT,
        next_fire_at=at,
    )
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_invoke_at",
                "args": {"prompt": "おはよう", "at": at.isoformat()},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.dispatcher_invoke_at)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="来週の9時に伝えて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.invoke_at.assert_called_once()
    invocation, called_at = m_service.invoke_at.call_args[0]
    assert called_at == at
    assert invocation["input"]["messages"][-1].content == "おはよう"
    # 観点2
    tool_message = result["messages"][2]
    assert "dispatch-1" in tool_message.content


async def test_dispatcher_invoke_at_02():
    """不正な日時文字列を渡した場合、Pydantic の検証で弾かれることを確認.

    観点1（R017）: DispatcherService.invoke_at() が呼ばれず、ToolMessage がエラー内容を
      含んで返ること
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_invoke_at",
                "args": {"prompt": "おはよう", "at": "明日の9時"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.dispatcher_invoke_at)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="明日の9時に伝えて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.invoke_at.assert_not_called()
    tool_message = result["messages"][2]
    assert tool_message.status == "error"


async def test_dispatcher_invoke_at_03():
    """DispatcherService.invoke_at() が ValueError を送出する場合の挙動を確認.

    観点1（R002・R017）: ツールが ValueError を再送出せず、エラーメッセージを文字列として返すこと
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.invoke_at.side_effect = ValueError("過去の日時は指定できません")
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_invoke_at",
                "args": {"prompt": "おはよう", "at": "2020-01-01T00:00:00"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.dispatcher_invoke_at)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="過去の日時に伝えて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点1
    tool_message = result["messages"][2]
    assert tool_message.status != "error"
    assert "過去の日時は指定できません" in tool_message.content


async def test_dispatcher_invoke_delay_01():
    """ツール dispatcher_invoke_delay が DispatcherService.invoke_delay() を正しく呼ぶことを確認.

    観点1（R017）: content・delay_value・delay_unit・interval_value・interval_unit を含む形で
      dispatcher_invoke_delay() が呼ばれ、invoke_delay() の戻り値（Dispatch）の文字列表現が
      LLM への ToolMessage に含まれること
    観点2（R017）: 引数を省略した場合は既定値（delay_value=10, delay_unit=SECONDS,
      interval_value=-1, interval_unit=MINUTES）が渡ること
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.invoke_delay.return_value = Dispatch(
        dispatch_id="dispatch-1",
        invocation={"input": {"messages": [HumanMessage(content="こんにちは")]}},
        interval_seconds=10,
        next_fire_at=datetime(2026, 7, 30, 9, 0, 0, tzinfo=UTC),
    )
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_invoke_delay",
                "args": {
                    "prompt": "こんにちは",
                    "delay_value": 5,
                    "delay_unit": "MINUTES",
                    "interval_value": 10,
                    "interval_unit": "SECONDS",
                },
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.dispatcher_invoke_delay)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="5分後に伝えて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.invoke_delay.assert_called_once()
    _, delay_value, delay_unit, interval_value, interval_unit = m_service.invoke_delay.call_args[0]
    assert (delay_value, delay_unit) == (5, IntervalUnit.MINUTES)
    assert (interval_value, interval_unit) == (10, IntervalUnit.SECONDS)
    tool_message = result["messages"][2]
    assert "dispatch-1" in tool_message.content

    # 試験準備: 引数省略
    m_service.invoke_delay.reset_mock()
    ai_msg_default = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_invoke_delay",
                "args": {"prompt": "今すぐ伝えて"},
                "id": "2",
                "type": "tool_call",
            }
        ],
    )
    agent_default = _make_agent(ai_msg_default, tools.dispatcher_invoke_delay)

    # 試験実施
    result_default = await agent_default.ainvoke(
        {"messages": [HumanMessage(content="今すぐ伝えて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点2
    _, delay_value, delay_unit, interval_value, interval_unit = m_service.invoke_delay.call_args[0]
    assert (delay_value, delay_unit) == (10, IntervalUnit.SECONDS)
    assert (interval_value, interval_unit) == (-1, IntervalUnit.MINUTES)
    tool_message_default = result_default["messages"][2]
    assert "dispatch-1" in tool_message_default.content


async def test_dispatcher_cancel_01():
    """ツール dispatcher_cancel が DispatcherService.cancel_dispatch() を呼ぶことを確認.

    観点1（R020）: dispatch_id とともに呼ばれること
    観点2（R020）: cancel_dispatch() の戻り値（True/False）と、その意味を説明する
      Description を含む文字列が返ること
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.cancel_dispatch.return_value = True
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_cancel",
                "args": {"dispatch_id": "dispatch-1"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.dispatcher_cancel)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="予定を消して")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.cancel_dispatch.assert_called_once_with("dispatch-1")
    # 観点2
    tool_message = result["messages"][2]
    assert "True" in tool_message.content
    assert "Description:" in tool_message.content

    # 試験準備: 見つからない場合
    m_service.cancel_dispatch.reset_mock(return_value=True)
    m_service.cancel_dispatch.return_value = False
    ai_msg_missing = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "dispatcher_cancel",
                "args": {"dispatch_id": "not-exist"},
                "id": "2",
                "type": "tool_call",
            }
        ],
    )
    agent_missing = _make_agent(ai_msg_missing, tools.dispatcher_cancel)

    # 試験実施
    result_missing = await agent_missing.ainvoke(
        {"messages": [HumanMessage(content="予定を消して")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点2
    tool_message_missing = result_missing["messages"][2]
    assert "False" in tool_message_missing.content
    assert "Description:" in tool_message_missing.content


async def test_dispatcher_list_01():
    """ツール dispatcher_list が DispatcherService.list_dispatch() の結果を整形して返すことを確認.

    観点1（R021）: 結果が dispatch_id・繰り返し間隔・次回発火時刻・メッセージ内容を含む
      文字列として返ること
    観点2（R021）: 予定が0件のときもその旨が返ること
    観点3（R021）: 100文字を超えるメッセージ内容が切り詰められること
    観点4（R021）: interval_seconds が ONE_SHOT の予定は "once"、それ以外は "every Ns" 等の
      人間可読な表現になり、-1s のような内部表現が出力に現れないこと
    観点5: ToolMessage.artifact が list_dispatch() の返り値と一致すること
    """
    # 試験準備
    long_content = "あ" * 150
    m_service = MagicMock(spec=DispatcherService)
    m_service.list_dispatch.return_value = [
        Dispatch(
            dispatch_id="dispatch-1",
            invocation={"input": {"messages": [HumanMessage(content=long_content)]}},
            interval_seconds=-1,
            next_fire_at=datetime(2026, 7, 30, 9, 0, 0, tzinfo=UTC),
        ),
        Dispatch(
            dispatch_id="dispatch-2",
            invocation={"input": {"messages": [HumanMessage(content="定期連絡")]}},
            interval_seconds=30,
            next_fire_at=datetime(2026, 7, 30, 9, 5, 0, tzinfo=UTC),
        ),
    ]
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "dispatcher_list", "args": {}, "id": "1", "type": "tool_call"}],
    )
    agent = _make_agent(ai_msg, tools.dispatcher_list)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="予定一覧を見せて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.list_dispatch.assert_called_once_with()
    tool_message = result["messages"][2]
    assert "dispatch-1" in tool_message.content
    assert "dispatch-2" in tool_message.content
    # 観点3
    assert "あ" * 101 not in tool_message.content
    assert "あ" * 100 in tool_message.content
    # 観点4
    assert "once" in tool_message.content
    assert "every 30s" in tool_message.content
    assert "-1s" not in tool_message.content
    # 観点5
    assert tool_message.artifact == m_service.list_dispatch.return_value

    # 試験準備: 0件
    m_service.list_dispatch.reset_mock(return_value=True)
    m_service.list_dispatch.return_value = []
    ai_msg_empty = AIMessage(
        content="",
        tool_calls=[{"name": "dispatcher_list", "args": {}, "id": "2", "type": "tool_call"}],
    )
    agent_empty = _make_agent(ai_msg_empty, tools.dispatcher_list)

    # 試験実施
    result_empty = await agent_empty.ainvoke(
        {"messages": [HumanMessage(content="予定一覧を見せて")]},
        config={"configurable": {"thread_id": "thread-1"}},
        context={"dispatcher_service": m_service},
    )

    # 結果検証
    # 観点2
    tool_message_empty = result_empty["messages"][2]
    assert tool_message_empty.content
    assert tool_message_empty.artifact == []


def _make_agent(tool_calls_msg: AIMessage, *tools):
    """テスト用エージェントを生成するヘルパー."""
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools))
