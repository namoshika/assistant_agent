import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import langchain.agents
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver
from assistant_agent.utils.context import CommonContext
from assistant_agent.utils.workflow import Agent, BroadcastPipe, LogWriter, MergePipe


class _DummyActiveEmitter(ActiveEmitter):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> _FakeChatModel:
        return self


class TestBroadcastPipe:
    def test_construct_01(self):
        """BroadcastPipe が src の発信を dst 全要素へ配信することを確認.

        観点1: src にメッセージを流すと dst の全要素の on_received() が呼ばれる
        """
        # 試験準備
        src = _DummyActiveEmitter()
        dst1, dst2 = MagicMock(spec=Receiver), MagicMock(spec=Receiver)
        invocation = AgentInvocation(input={"messages": [HumanMessage(content="hello")]})

        # 試験実施
        BroadcastPipe(src, [dst1, dst2])
        src.emit(invocation)

        # 結果検証
        # 観点1
        dst1.on_received.assert_called_once_with(invocation)
        dst2.on_received.assert_called_once_with(invocation)


class TestMergePipe:
    def test_construct_01(self):
        """MergePipe が src 複数の発信を dst 1つへ集約することを確認.

        観点1: src1・src2 それぞれにメッセージを流すと、両方とも dst.on_received() が呼ばれる
        """
        # 試験準備
        src1, src2 = _DummyActiveEmitter(), _DummyActiveEmitter()
        dst = MagicMock(spec=Receiver)
        invocation1 = AgentInvocation(input={"messages": [HumanMessage(content="hello1")]})
        invocation2 = AgentInvocation(input={"messages": [HumanMessage(content="hello2")]})

        # 試験実施
        MergePipe([src1, src2], dst)
        src1.emit(invocation1)
        src2.emit(invocation2)

        # 結果検証
        # 観点1
        dst.on_received.assert_any_call(invocation1)
        dst.on_received.assert_any_call(invocation2)
        assert dst.on_received.call_count == 2


class TestAgent:
    async def test_receive_01(self):
        """BroadcastPipe から受けた新着に実グラフが応答し、自身の購読者へ配信されることを確認.

        観点1: BroadcastPipe で接続した発信元から新着を流すと、
            実グラフの応答が Agent 自身の購読者へ配信される
        """
        # 試験準備
        lc_agent = langchain.agents.create_agent(
            model=_FakeChatModel(messages=iter([AIMessage(content="reply1")])),
            tools=[],
            system_prompt="test",
        )
        source = _DummyActiveEmitter()
        agent = Agent(lc_agent, context={})
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施
        agent.start()
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        received.on_received.assert_called_once()
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["input"]["messages"][-1].content == "reply1"

    async def test_receive_02(self):
        """Agent 同士が連鎖して応答し、コンストラクタの thread_id・context が使われることを確認.

        観点1: BroadcastPipe(src, [agent1])・BroadcastPipe(agent1, [agent2]) で連結し、
            agent1 の応答が agent2 の入力となり agent2 の応答が配信される
        観点2: agent2 のグラフ実行時に thread_id を指定した config と、context が渡される事
        """
        # 試験準備
        lc_agent1 = langchain.agents.create_agent(
            model=_FakeChatModel(messages=iter([AIMessage(content="reply1")])),
            tools=[],
            system_prompt="test",
        )
        lc_agent2 = MagicMock(spec=CompiledStateGraph)
        lc_agent2.ainvoke = AsyncMock(
            return_value=GraphOutput(value={"messages": [AIMessage(content="reply2")]})
        )
        source = _DummyActiveEmitter()
        agent1 = Agent(lc_agent1, context={})
        context: CommonContext = {"sample_retriever": "dummy"}  # pyright: ignore[reportAssignmentType]
        agent2 = Agent(lc_agent2, context=context, thread_id="fixed-thread-id")
        BroadcastPipe(source, [agent1])
        BroadcastPipe(agent1, [agent2])
        received = MagicMock(spec=Receiver)
        agent2.receiver = received

        # 試験実施
        agent1.start()
        agent2.start()
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        received.on_received.assert_called_once()
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["input"]["messages"][-1].content == "reply2"
        # 観点2
        _, kwargs = lc_agent2.ainvoke.call_args
        assert kwargs["config"]["configurable"]["thread_id"] == "fixed-thread-id"
        assert kwargs["context"] == context

    async def test_receive_03(self, caplog: pytest.LogCaptureFixture):
        """ainvoke() 実行中に例外が発生しても _consume() が停止しないことを確認.

        観点1: 例外発生時、logger.error() でトレース情報が記録されること
        観点2: 例外発生後も後続メッセージが処理されること（_consume() のループが継続する）
        """
        # 試験準備
        lc_agent = MagicMock(spec=CompiledStateGraph)
        lc_agent.ainvoke = AsyncMock(
            side_effect=[
                KeyError("NOT_FOUND_CONTEXT"),
                GraphOutput(value={"messages": [AIMessage(content="reply after error")]}),
            ]
        )
        source = _DummyActiveEmitter()
        agent = Agent(lc_agent, context={}, thread_id="fixed-thread-id")
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施
        with caplog.at_level(logging.ERROR, logger="assistant_agent.utils.workflow"):
            agent.start()
            source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}))
            await asyncio.sleep(1)
            source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi again")]}))
            await asyncio.sleep(1)

        # 結果検証
        # 観点1
        assert "KeyError" in caplog.text
        assert "fixed-thread-id" in caplog.text
        # 観点2
        received.on_received.assert_called_once()
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["input"]["messages"][-1].content == "reply after error"

    async def test_start_01(self):
        """start()/stop() による処理のライフサイクルを確認.

        観点1: BroadcastPipe で接続した2つの Agent を start() し、
            新着を流すと両方が処理してそれぞれの購読者へ配信すること
        観点2: stop() 中は新着を流しても処理されないこと（キューには残る）
        観点3: 再 start() すると停止中に届いたメッセージも含めて処理されること。
            また start() を2回呼んでも二重に処理されないこと
        """
        # 試験準備
        lc_agent1 = langchain.agents.create_agent(
            model=_FakeChatModel(
                messages=iter(
                    [
                        AIMessage(content="reply1a"),
                        AIMessage(content="reply1b"),
                    ]
                )
            ),
            tools=[],
            system_prompt="test",
        )
        lc_agent2 = langchain.agents.create_agent(
            model=_FakeChatModel(messages=iter([AIMessage(content="reply2")])),
            tools=[],
            system_prompt="test",
        )
        source = _DummyActiveEmitter()
        agent1 = Agent(lc_agent1, context={})
        agent2 = Agent(lc_agent2, context={})
        BroadcastPipe(source, [agent1, agent2])
        received1 = MagicMock(spec=Receiver)
        received2 = MagicMock(spec=Receiver)
        agent1.receiver = received1
        agent2.receiver = received2

        # 試験実施
        agent1.start()
        agent2.start()
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        received1.on_received.assert_called_once()
        received2.on_received.assert_called_once()

        # 試験実施: stop 中は配信されない（キューには残る）
        agent1.stop()
        agent2.stop()
        await asyncio.sleep(1)
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="after stop")]}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点2
        received1.on_received.assert_called_once()
        received2.on_received.assert_called_once()

        # 試験実施: start を2回呼んでも処理は1回だけ
        agent1.start()
        agent1.start()
        await asyncio.sleep(1)

        # 結果検証: stop 中に届いた "after stop" が処理され、配信は計2回
        # 観点3
        assert received1.on_received.call_count == 2


class TestLogWriter:
    def test_on_received_01(self, caplog: pytest.LogCaptureFixture):
        """on_received() がメッセージ本文を logging 経由で記録することを確認.

        観点1: on_received() を呼ぶと、logging.getLogger(__name__) にメッセージ本文が INFO レベルで記録されること
        観点2: 複数回呼ぶと、両方の内容が記録に残ること
        """  # noqa: E501
        # 試験準備
        writer = LogWriter()

        # 試験実施
        with caplog.at_level(logging.DEBUG, logger="assistant_agent.utils.workflow"):
            writer.on_received(AgentInvocation(input={"messages": [HumanMessage(content="hello")]}))
            writer.on_received(AgentInvocation(input={"messages": [HumanMessage(content="world")]}))

        # 結果検証
        # 観点1
        assert "hello" in caplog.text
        # 観点2
        assert "world" in caplog.text
