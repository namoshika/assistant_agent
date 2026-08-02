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
from assistant_agent.utils.workflow import (
    Agent,
    BroadcastPipe,
    LogWriter,
    MergePipe,
    SyncRequestChannel,
)


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
        invocation = AgentInvocation(
            input={"messages": [HumanMessage(content="hello")]}, context={}
        )

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
        invocation1 = AgentInvocation(
            input={"messages": [HumanMessage(content="hello1")]}, context={}
        )
        invocation2 = AgentInvocation(
            input={"messages": [HumanMessage(content="hello2")]}, context={}
        )

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
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={}))
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
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={}))
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
        """ainvoke() 実行中に例外が発生しても _consume() が停止せず、エラー内容が emit されることを確認.

        観点1: 例外発生時、logger.error() でトレース情報が記録されること
        観点2: 例外発生時もエラー内容を積んだ AgentInvocation が emit されること
        観点3: 例外発生後も後続メッセージが処理されること（_consume() のループが継続する）
        """  # noqa: E501
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
            source.emit(
                AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={})
            )
            await asyncio.sleep(1)
            source.emit(
                AgentInvocation(input={"messages": [HumanMessage(content="hi again")]}, context={})
            )
            await asyncio.sleep(1)

        # 結果検証
        # 観点1
        assert "KeyError" in caplog.text
        assert "fixed-thread-id" in caplog.text
        # 観点2
        assert received.on_received.call_count == 2
        error_invocation: AgentInvocation = received.on_received.call_args_list[0][0][0]
        assert isinstance(error_invocation["input"]["messages"][-1], AIMessage)
        # 観点3
        result_invocation: AgentInvocation = received.on_received.call_args_list[1][0][0]
        assert result_invocation["input"]["messages"][-1].content == "reply after error"

    async def test_receive_04(self):
        """入力の context がサービス群の context とマージされて ainvoke() へ渡ることを確認.

        観点1: 入力に context={"request_id": "xxx"} を含めて emit すると、ainvoke() へ渡る
            context にサービス群の値と request_id の両方が含まれること
        観点2: emit される応答の AgentInvocation["context"] に入力側の context がそのまま
            積まれること
        """
        # 試験準備
        lc_agent = MagicMock(spec=CompiledStateGraph)
        lc_agent.ainvoke = AsyncMock(
            return_value=GraphOutput(value={"messages": [AIMessage(content="reply")]})
        )
        source = _DummyActiveEmitter()
        context: CommonContext = {"sample_retriever": "dummy"}  # pyright: ignore[reportAssignmentType]
        agent = Agent(lc_agent, context=context)
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施
        agent.start()
        source.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content="hi")]},
                context={"request_id": "req-1"},
            )
        )
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        _, kwargs = lc_agent.ainvoke.call_args
        assert kwargs["context"] == {"sample_retriever": "dummy", "request_id": "req-1"}
        # 観点2
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["context"] == {"request_id": "req-1"}

    async def test_receive_05(self):
        """ainvoke() のタイムアウト・例外時にエラー内容を積んだ応答が emit されることを確認.

        観点1: ainvoke() が context.timeout_seconds 以内に完了しない場合、AIMessage
            （エラー内容）を積んだ AgentInvocation が context を引き継いだ状態で emit される
        観点2: context.timeout_seconds を指定した場合、TIMEOUT_SECONDS_DEFAULT ではなく
            指定した秒数でタイムアウトすること
        観点3: ainvoke() がタイムアウト以外の例外を送出した場合も、AIMessage（エラー内容）を
            積んだ AgentInvocation が context を引き継いだ状態で emit されること
        """

        async def _sleep_forever(**_: Any) -> GraphOutput:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        lc_agent = MagicMock(spec=CompiledStateGraph)
        lc_agent.ainvoke = AsyncMock(side_effect=_sleep_forever)
        source = _DummyActiveEmitter()
        agent = Agent(lc_agent, context={})
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施
        agent.start()
        source.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content="hi")]},
                context={"request_id": "req-timeout", "timeout_seconds": 1},
            )
        )
        await asyncio.sleep(3)

        # 結果検証
        # 観点1・観点2
        received.on_received.assert_called_once()
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["context"] == {"request_id": "req-timeout", "timeout_seconds": 1}
        msg_out = result_invocation["input"]["messages"][-1]
        assert isinstance(msg_out, AIMessage)

        # 試験準備: タイムアウト以外の例外
        lc_agent2 = MagicMock(spec=CompiledStateGraph)
        lc_agent2.ainvoke = AsyncMock(side_effect=KeyError("boom"))
        agent2 = Agent(lc_agent2, context={})
        BroadcastPipe(source, [agent2])
        received2 = MagicMock(spec=Receiver)
        agent2.receiver = received2

        # 試験実施
        agent2.start()
        source.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content="hi")]},
                context={"request_id": "req-error"},
            )
        )
        await asyncio.sleep(1)

        # 結果検証
        # 観点3
        received2.on_received.assert_called_once()
        result_invocation2: AgentInvocation = received2.on_received.call_args[0][0]
        assert result_invocation2["context"] == {"request_id": "req-error"}
        assert isinstance(result_invocation2["input"]["messages"][-1], AIMessage)

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
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        received1.on_received.assert_called_once()
        received2.on_received.assert_called_once()

        # 試験実施: stop 中は配信されない（キューには残る）
        agent1.stop()
        agent2.stop()
        await asyncio.sleep(1)
        source.emit(
            AgentInvocation(input={"messages": [HumanMessage(content="after stop")]}, context={})
        )
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


class TestSyncRequestChannel:
    async def test_emit_and_wait_01(self):
        """emit_and_wait() が content・channel_name, request_id を持つ AgentInvocation を emit することを確認.

        観点1: emit する AgentInvocation の input.messages[-1].content に引数の channel_name と content が含まれること
        観点2: emit する AgentInvocation の context.request_id が一意な値であること
        """  # noqa: E501
        # 試験準備
        channel = SyncRequestChannel()
        received = MagicMock(spec=Receiver)
        channel.receiver = received

        # 試験実施
        task1 = asyncio.ensure_future(
            channel.emit_and_wait("hello", channel_name="Test Channel", timeout_seconds=1)
        )
        task2 = asyncio.ensure_future(
            channel.emit_and_wait("world", channel_name="Test Channel", timeout_seconds=1)
        )
        await asyncio.sleep(0.2)

        # 結果検証
        # 観点1
        invocation1: AgentInvocation = received.on_received.call_args_list[0][0][0]
        invocation2: AgentInvocation = received.on_received.call_args_list[1][0][0]
        content1 = invocation1["input"]["messages"][-1].content
        content2 = invocation2["input"]["messages"][-1].content
        assert "Test Channel" in content1
        assert "hello" in content1
        assert "Test Channel" in content2
        assert "world" in content2
        # 観点2
        assert invocation1["context"]["request_id"] != invocation2["context"]["request_id"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        # 後始末
        for task in (task1, task2):
            task.cancel()
            with pytest.raises((asyncio.CancelledError, TimeoutError)):
                await task

    async def test_emit_and_wait_02(self):
        """emit_and_wait() の応答解決・未知の request_id の扱いを確認.

        観点1: emit_and_wait() 呼び出し後、対応する context.request_id を積んだ AgentInvocation を on_received() に渡すと await が解決し、戻り値の BaseMessage が応答内容と一致すること
        観点2: 未知の request_id（_pending に無い値）を持つ AgentInvocation を on_received() に渡しても例外を送出しないこと
        """  # noqa: E501
        # 試験準備
        channel = SyncRequestChannel()
        received = MagicMock(spec=Receiver)
        channel.receiver = received

        # 試験実施
        task = asyncio.ensure_future(channel.emit_and_wait("hello", channel_name="Test Channel"))
        await asyncio.sleep(0.05)
        invocation: AgentInvocation = received.on_received.call_args[0][0]
        request_id = invocation["context"]["request_id"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        channel.on_received(
            AgentInvocation(
                input={"messages": [AIMessage(content="reply")]},
                context={"request_id": request_id},
            )
        )
        result = await task

        # 結果検証
        # 観点1
        assert result.content == "reply"

        # 試験実施・結果検証: 未知の request_id
        # 観点2
        channel.on_received(
            AgentInvocation(
                input={"messages": [AIMessage(content="reply")]},
                context={"request_id": "unknown"},
            )
        )

    async def test_emit_and_wait_03(self):
        """timeout_seconds 経過時・stop() 時の待機中断を確認.

        観点1: 対応する応答が来ないまま timeout_seconds が経過すると TimeoutError を送出すること
        観点2: stop() を呼び出すと、待機中の emit_and_wait() が CancelledError で終了すること
        """
        # 試験準備
        channel = SyncRequestChannel()
        channel.receiver = MagicMock(spec=Receiver)

        # 試験実施・結果検証
        # 観点1
        with pytest.raises(TimeoutError):
            await channel.emit_and_wait("hello", channel_name="Test Channel", timeout_seconds=0)

        # 試験準備: stop() によるキャンセル
        task = asyncio.ensure_future(
            channel.emit_and_wait("hello", channel_name="Test Channel", timeout_seconds=10)
        )
        await asyncio.sleep(0.05)

        # 試験実施
        channel.stop()

        # 結果検証
        # 観点2
        with pytest.raises(asyncio.CancelledError):
            await task


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
            writer.on_received(
                AgentInvocation(input={"messages": [HumanMessage(content="hello")]}, context={})
            )
            writer.on_received(
                AgentInvocation(input={"messages": [HumanMessage(content="world")]}, context={})
            )

        # 結果検証
        # 観点1
        assert "hello" in caplog.text
        # 観点2
        assert "world" in caplog.text
