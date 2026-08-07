from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

import assistant_agent.tools.dispatcher as tools
from assistant_agent.services.dispatcher import Dispatch, DispatcherService
from assistant_agent.tools.dispatcher import DispatcherContext


@pytest.mark.integration
async def test_dispatcher_invoke_at_01(llm: BaseChatModel) -> None:
    """実際の LLM から dispatcher_invoke_at を呼び出せるか確認.

    観点1: LLM が content・at を含む tool call を生成し invoke_at が呼ばれる
    観点2: invoke_at の戻り値（Dispatch）の文字列表現が ToolMessage.content に含まれる
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.invoke_at.return_value = Dispatch(
        dispatch_id="dispatch-1",
        agent_id="test-agent",
        prompt="おはよう",
        interval_seconds=DispatcherService.ONE_SHOT,
        run_at=datetime.now(UTC),
    )
    agent = create_agent(
        model=llm, tools=[tools.dispatcher_invoke_at], context_schema=DispatcherContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="2026年8月1日9時に「おはよう」と伝えて")]},
        context=cast(
            DispatcherContext, {"dispatcher_service": m_service, "agent_id": "test-agent"}
        ),
    )

    # 結果検証
    # 観点1
    m_service.invoke_at.assert_called_once()
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "dispatch-1" in tool_msg.content


@pytest.mark.integration
async def test_dispatcher_invoke_delay_01(llm: BaseChatModel) -> None:
    """実際の LLM から dispatcher_invoke_delay を呼び出せるか確認.

    観点1: LLM が content を含む tool call を生成し invoke_delay が呼ばれる
    観点2: invoke_delay の戻り値（Dispatch）の文字列表現が ToolMessage.content に含まれる
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.invoke_delay.return_value = Dispatch(
        dispatch_id="dispatch-2",
        agent_id="test-agent",
        prompt="こんにちは",
        interval_seconds=DispatcherService.ONE_SHOT,
        run_at=datetime.now(UTC),
    )
    agent = create_agent(
        model=llm, tools=[tools.dispatcher_invoke_delay], context_schema=DispatcherContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="10秒後に「こんにちは」と伝えて")]},
        context=cast(
            DispatcherContext, {"dispatcher_service": m_service, "agent_id": "test-agent"}
        ),
    )

    # 結果検証
    # 観点1
    m_service.invoke_delay.assert_called_once()
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "dispatch-2" in tool_msg.content


@pytest.mark.integration
async def test_dispatcher_cancel_01(llm: BaseChatModel) -> None:
    """実際の LLM から dispatcher_cancel を呼び出せるか確認.

    観点1: LLM が dispatch_id を含む tool call を生成し cancel_dispatch が agent_id とともに
      呼ばれる
    観点2: cancel_dispatch の戻り値が ToolMessage.content に含まれる
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.cancel_dispatch.return_value = True
    agent = create_agent(
        model=llm, tools=[tools.dispatcher_cancel], context_schema=DispatcherContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="dispatch-3 の予定をキャンセルして")]},
        context=cast(
            DispatcherContext, {"dispatcher_service": m_service, "agent_id": "test-agent"}
        ),
    )

    # 結果検証
    # 観点1
    assert m_service.cancel_dispatch.call_args.args == ("test-agent", "dispatch-3")
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "True" in tool_msg.content


@pytest.mark.integration
async def test_dispatcher_list_01(llm: BaseChatModel) -> None:
    """実際の LLM から dispatcher_list を呼び出せるか確認.

    観点1: LLM が tool call を生成し list_dispatch が呼ばれる
    観点2: list_dispatch の戻り値（Dispatch）の文字列表現が ToolMessage.content に含まれる
    """
    # 試験準備
    m_service = MagicMock(spec=DispatcherService)
    m_service.list_dispatch.return_value = [
        Dispatch(
            dispatch_id="dispatch-4",
            agent_id="test-agent",
            prompt="定期連絡",
            interval_seconds=DispatcherService.ONE_SHOT,
            run_at=datetime.now(UTC),
        )
    ]
    agent = create_agent(model=llm, tools=[tools.dispatcher_list], context_schema=DispatcherContext)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="登録済みの予定を一覧で見せて")]},
        context=cast(
            DispatcherContext, {"dispatcher_service": m_service, "agent_id": "test-agent"}
        ),
    )

    # 結果検証
    # 観点1
    m_service.list_dispatch.assert_called_once()
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "dispatch-4" in tool_msg.content
