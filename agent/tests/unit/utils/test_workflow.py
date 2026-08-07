import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import langchain.agents
import pytest
from deepagents.backends.store import StoreBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import GraphOutput
from pytest_mock import MockerFixture

from assistant_agent.utils.absclass import (
    ActiveEmitter,
    AgentInvocation,
    Receiver,
    RolloverStrategy,
)
from assistant_agent.utils.context import CommonContext
from assistant_agent.utils.workflow import (
    THREAD_ROLLOVER_INTERVAL_DAYS,
    Agent,
    BroadcastPipe,
    DefaultRolloverStrategy,
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


class _NoopRolloverStrategy(RolloverStrategy):
    async def invoke(self, lc_agent, agent_id, thread_id, config):
        return thread_id, config


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


@pytest.fixture
def fixed_thread_id() -> str:
    """ロールオーバー対象と判定されない（生成直後扱いの）固定 thread_id を返す."""
    return f"test-agent:{uuid.uuid7()}"


@pytest.fixture
def old_thread_id() -> str:
    """ロールオーバー対象と判定される（THREAD_ROLLOVER_INTERVAL_DAYS 以上前の）thread_id を返す."""
    # uuid.uuid7() はモジュールグローバルな単調性保証機構（前回生成時刻を記憶し、
    # それより古い時刻の生成要求を直前時刻+1ms へ補正する）を持つため、
    # time.time_ns() のモック化では過去日時を埋め込めない。そのため、通常どおり
    # 生成した UUID のタイムスタンプ部分（上位80ビット）のみを過去日時へ差し替える。
    old_time = datetime.now(UTC) - timedelta(days=THREAD_ROLLOVER_INTERVAL_DAYS + 1)
    old_ts_ms = int(old_time.timestamp() * 1000)
    base_uuid = uuid.uuid7()
    old_uuid_int = (base_uuid.int & ((1 << 80) - 1)) | (old_ts_ms << 80)
    return f"test-agent:{uuid.UUID(int=old_uuid_int)}"


class TestAgent:
    async def test_receive_01(self):
        """BroadcastPipe から受けた新着に実グラフが応答し、自身の購読者へ配信されることを確認.

        観点1: BroadcastPipe で接続した発信元から新着を流すと、実グラフの応答が Agent 自身の購読者へ配信される
        観点2: 応答の content に reasoning ブロックが含まれても例外にならず、そのまま配信されること
        """  # noqa: E501
        # 試験準備
        reasoning_block = {"type": "reasoning", "id": "rs_test", "summary": []}
        text_block = {"type": "text", "text": "reply2"}
        lc_agent = langchain.agents.create_agent(
            model=_FakeChatModel(
                messages=iter(
                    [
                        AIMessage(content="reply1"),
                        AIMessage(content=[reasoning_block, text_block]),
                    ]
                )
            ),
            tools=[],
            system_prompt="test",
        )
        source = _DummyActiveEmitter()
        agent = Agent(
            lc_agent,
            context={},
            agent_id="test-agent",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
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

        # 試験実施: reasoning ブロックを含む応答
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点2
        assert received.on_received.call_count == 2
        result_invocation2: AgentInvocation = received.on_received.call_args_list[1][0][0]
        assert result_invocation2["input"]["messages"][-1].content == [
            reasoning_block,
            text_block,
        ]

    async def test_receive_02(self, fixed_thread_id: str):
        """Agent 同士が連鎖して応答し、コンストラクタの thread_id・context が使われることを確認.

        観点1: BroadcastPipe(src, [agent1])・BroadcastPipe(agent1, [agent2]) で連結し、
            agent1 の応答が agent2 の入力となり agent2 の応答が配信される
        観点2: agent2 のグラフ実行時に thread_id を指定した config と、agent_id を含む context が
            渡される事
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
        agent1 = Agent(
            lc_agent1,
            context={},
            agent_id="test-agent",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
        context: CommonContext = {"sample_retriever": "dummy"}  # pyright: ignore[reportAssignmentType]
        agent2 = Agent(
            lc_agent2,
            context=context,
            agent_id="test-agent",
            thread_id=fixed_thread_id,
            rollover_strategy=_NoopRolloverStrategy(),
        )
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
        assert kwargs["config"]["configurable"]["thread_id"] == fixed_thread_id
        assert kwargs["context"] == context | {"agent_id": "test-agent"}

    async def test_receive_03(self):
        """入力の context がサービス群の context とマージされて ainvoke() へ渡ることを確認.

        観点1: 入力に context={"request_id": "xxx"} を含めて emit すると、ainvoke() へ渡る
            context にサービス群の値、request_id、自身の agent_id が含まれること
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
        agent = Agent(
            lc_agent,
            context=context,
            agent_id="test-agent",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
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
        assert kwargs["context"] == {
            "sample_retriever": "dummy",
            "request_id": "req-1",
            "agent_id": "test-agent",
        }
        # 観点2
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["context"] == {"request_id": "req-1"}

    async def test_receive_04(self, fixed_thread_id: str):
        """rollover_strategy を呼び出し、その戻り値の thread_id, config を使うことを確認.

        観点1: rollover_strategy が呼ばれること
        観点2: rollover_strategy が返した新しい thread_id, config が ainvoke() に使われること
        """
        # 試験準備
        lc_agent = MagicMock(spec=CompiledStateGraph)
        lc_agent.ainvoke = AsyncMock(
            return_value=GraphOutput(value={"messages": [AIMessage(content="reply")]})
        )
        new_thread_id = "test-agent:new-thread"
        new_config = {"configurable": {"thread_id": new_thread_id}}
        rollover_strategy = MagicMock(spec=RolloverStrategy)
        rollover_strategy.invoke = AsyncMock(return_value=(new_thread_id, new_config))
        source = _DummyActiveEmitter()
        agent = Agent(
            lc_agent,
            context={},
            agent_id="test-agent",
            thread_id=fixed_thread_id,
            rollover_strategy=rollover_strategy,
        )
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施
        agent.start()
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        rollover_strategy.invoke.assert_called_once()
        call_args = rollover_strategy.invoke.call_args.args
        assert call_args[0] is lc_agent
        assert call_args[1] == "test-agent"
        assert call_args[2] == fixed_thread_id
        assert call_args[3] == {"configurable": {"thread_id": fixed_thread_id}}
        # 観点2
        _, kwargs = lc_agent.ainvoke.call_args
        assert kwargs["config"] == new_config

    async def test_receive_05(self, caplog: pytest.LogCaptureFixture, fixed_thread_id: str):
        """ainvoke() 実行中に例外が発生しても _consume() が停止せず、エラー内容が emit されることを確認.

        観点1: 例外発生時、logger.error() でトレース情報が記録されること
        観点2: 例外発生時もエラー内容を積んだ AgentInvocation が context を引き継いだ状態で emit されること
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
        agent = Agent(
            lc_agent,
            context={},
            agent_id="test-agent",
            thread_id=fixed_thread_id,
            rollover_strategy=_NoopRolloverStrategy(),
        )
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施
        with caplog.at_level(logging.ERROR, logger="assistant_agent.utils.workflow"):
            agent.start()
            source.emit(
                AgentInvocation(
                    input={"messages": [HumanMessage(content="hi")]},
                    context={"request_id": "req-error"},
                )
            )
            await asyncio.sleep(1)
            source.emit(
                AgentInvocation(input={"messages": [HumanMessage(content="hi again")]}, context={})
            )
            await asyncio.sleep(1)

        # 結果検証
        # 観点1
        assert "KeyError" in caplog.text
        assert fixed_thread_id in caplog.text
        # 観点2
        assert received.on_received.call_count == 2
        error_invocation: AgentInvocation = received.on_received.call_args_list[0][0][0]
        assert isinstance(error_invocation["input"]["messages"][-1], AIMessage)
        assert error_invocation["context"] == {"request_id": "req-error"}
        # 観点3
        result_invocation: AgentInvocation = received.on_received.call_args_list[1][0][0]
        assert result_invocation["input"]["messages"][-1].content == "reply after error"

    async def test_receive_06(self):
        """ainvoke() が context.timeout_seconds 以内に完了しない場合の挙動を確認.

        観点1: AIMessage（エラー内容）が context を引き継いだ状態で emit されること
        観点2: TIMEOUT_SECONDS_DEFAULT（300秒）ではなく context.timeout_seconds に指定した秒数でタイムアウトすること
        """  # noqa: E501

        async def _sleep_forever(**_: Any) -> GraphOutput:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        lc_agent = MagicMock(spec=CompiledStateGraph)
        lc_agent.ainvoke = AsyncMock(side_effect=_sleep_forever)
        source = _DummyActiveEmitter()
        agent = Agent(
            lc_agent,
            context={},
            agent_id="test-agent",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
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
        # 観点1～2
        received.on_received.assert_called_once()
        result_invocation: AgentInvocation = received.on_received.call_args[0][0]
        assert result_invocation["context"] == {"request_id": "req-timeout", "timeout_seconds": 1}
        msg_out = result_invocation["input"]["messages"][-1]
        assert isinstance(msg_out, AIMessage)

    async def test_receive_07(self):
        """受信した context["agent_id"] によるフィルタを確認.

        観点1: context["agent_id"] が自身の agent_id と不一致の場合、ainvoke() が呼ばれず
            応答も配信されないこと
        観点2: context["agent_id"] が自身の agent_id と一致する場合、従来どおり処理されること
        観点3: context に agent_id を含まない場合、従来どおり処理されること
        """
        # 試験準備
        lc_agent = MagicMock(spec=CompiledStateGraph)
        lc_agent.ainvoke = AsyncMock(
            return_value=GraphOutput(value={"messages": [AIMessage(content="reply")]})
        )
        source = _DummyActiveEmitter()
        agent = Agent(
            lc_agent,
            context={},
            agent_id="test-agent",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
        BroadcastPipe(source, [agent])
        received = MagicMock(spec=Receiver)
        agent.receiver = received

        # 試験実施: 不一致の agent_id
        agent.start()
        source.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content="hi")]},
                context={"agent_id": "other-agent"},
            )
        )
        await asyncio.sleep(1)

        # 結果検証
        # 観点1
        lc_agent.ainvoke.assert_not_called()
        received.on_received.assert_not_called()

        # 試験実施: 一致する agent_id
        source.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content="hi")]},
                context={"agent_id": "test-agent"},
            )
        )
        await asyncio.sleep(1)

        # 結果検証
        # 観点2
        lc_agent.ainvoke.assert_called_once()
        received.on_received.assert_called_once()

        # 試験実施: agent_id を含まない
        source.emit(AgentInvocation(input={"messages": [HumanMessage(content="hi")]}, context={}))
        await asyncio.sleep(1)

        # 結果検証
        # 観点3
        assert lc_agent.ainvoke.call_count == 2
        assert received.on_received.call_count == 2

    async def test_start_01(self):
        """start()/stop() による処理のライフサイクルを確認.

        観点1: BroadcastPipe で接続した2つの Agent を start() し、新着を流すと両方が処理してそれぞれの購読者へ配信すること
        観点2: stop() 中は新着を流しても処理されないこと（キューには残る）
        観点3: 再 start() すると停止中に届いたメッセージも含めて処理されること。また start() を2回呼んでも二重に処理されないこと
        """  # noqa: E501
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
        agent1 = Agent(
            lc_agent1,
            context={},
            agent_id="test-agent-1",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
        agent2 = Agent(
            lc_agent2,
            context={},
            agent_id="test-agent-2",
            thread_id=None,
            rollover_strategy=_NoopRolloverStrategy(),
        )
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


class TestDefaultRolloverStrategy:
    def test_should_rollover_01(self, fixed_thread_id: str, old_thread_id: str):
        """should_rollover() が thread_id 生成時刻からの経過日数で判定することを確認.

        観点1: THREAD_ROLLOVER_INTERVAL_DAYS 未満の経過では False を返すこと
        観点2: THREAD_ROLLOVER_INTERVAL_DAYS 以上の経過では True を返すこと
        観点3: 想定外の形式の thread_id では True を返すこと
        """
        # 試験実施、結果検証
        # 観点1
        assert DefaultRolloverStrategy.should_rollover(fixed_thread_id) is False
        # 観点2
        assert DefaultRolloverStrategy.should_rollover(old_thread_id) is True
        # 観点3
        assert DefaultRolloverStrategy.should_rollover("not-a-valid-thread-id") is True

    async def test_invoke_01(self, old_thread_id: str):
        """ロールオーバー条件を満たすターンで thread_id 切替・要約差込・生ログ退避が行われるか確認.

        観点1: thread_id がロールオーバー前と異なる値へ切り替わること
        観点2: 要約生成が行われ、生成された要約が新 thread の messages へ差し込まれること
        観点3: 旧 thread の全メッセージが backend へ退避され、read_file で参照できること
        観点4: ロールオーバー条件を満たさないターンでは、渡された thread_id, config がそのまま返り、
            新 thread の checkpoint が作られないこと
        """
        # 試験準備
        store = InMemoryStore()
        backend = StoreBackend(store=store, namespace=lambda _rt: ("test-agent", "filesystem"))
        llm = _FakeChatModel(messages=iter([AIMessage(content="summary text")]))
        strategy = DefaultRolloverStrategy(llm, backend)

        lc_agent = langchain.agents.create_agent(
            model=llm, tools=[], system_prompt="test", checkpointer=InMemorySaver()
        )
        old_config: RunnableConfig = {"configurable": {"thread_id": old_thread_id}}
        await lc_agent.aupdate_state(
            old_config,
            {"messages": [HumanMessage(content="old question"), AIMessage(content="old answer")]},
        )

        # 試験実施
        new_thread_id, new_config = await strategy.invoke(
            lc_agent, "test-agent", old_thread_id, old_config
        )

        # 結果検証
        # 観点1
        assert new_thread_id != old_thread_id
        assert new_thread_id.startswith("test-agent:")
        assert new_config.get("configurable", {}).get("thread_id") == new_thread_id
        # 観点2
        new_snapshot = await lc_agent.aget_state(new_config)
        summary_message = new_snapshot.values["messages"][0]
        assert isinstance(summary_message, HumanMessage)
        assert "summary text" in summary_message.content
        # 観点3
        history = await backend.aread(f"/conversation_history/{old_thread_id}.md")
        assert history.file_data is not None
        assert "old question" in history.file_data["content"]
        assert "old answer" in history.file_data["content"]

        # 試験準備: ロールオーバー条件を満たさない thread_id
        recent_thread_id = f"test-agent:{uuid.uuid7()}"
        recent_config: RunnableConfig = {"configurable": {"thread_id": recent_thread_id}}

        # 試験実施
        result_thread_id, result_config = await strategy.invoke(
            lc_agent, "test-agent", recent_thread_id, recent_config
        )

        # 結果検証
        # 観点4
        assert result_thread_id == recent_thread_id
        assert result_config == recent_config
        recent_snapshot = await lc_agent.aget_state(recent_config)
        assert recent_snapshot.values == {}

    async def test_invoke_02(self, old_thread_id: str):
        """Messages が空の状態でロールオーバー条件を満たした場合、要約生成をスキップすることを確認.

        観点1: 新 thread の checkpoint に要約メッセージが差し込まれずに thread_id が切り替わること
        """
        # 試験準備
        store = InMemoryStore()
        backend = StoreBackend(store=store, namespace=lambda _rt: ("test-agent", "filesystem"))
        llm = _FakeChatModel(messages=iter([]))
        strategy = DefaultRolloverStrategy(llm, backend)

        lc_agent = langchain.agents.create_agent(
            model=llm, tools=[], system_prompt="test", checkpointer=InMemorySaver()
        )
        old_config: RunnableConfig = {"configurable": {"thread_id": old_thread_id}}

        # 試験実施
        new_thread_id, new_config = await strategy.invoke(
            lc_agent, "test-agent", old_thread_id, old_config
        )

        # 結果検証
        # 観点1
        assert new_thread_id != old_thread_id
        new_snapshot = await lc_agent.aget_state(new_config)
        assert new_snapshot.values == {}

    async def test_invoke_03(self, mocker: MockerFixture, old_thread_id: str):
        """要約生成が失敗しても thread_id の世代交代自体は継続することを確認.

        観点1: 要約生成が例外を送出しても新しい thread_id, config が返ること
        """
        # 試験準備
        backend = mocker.Mock(spec=StoreBackend)
        backend.awrite = AsyncMock()
        llm = mocker.Mock(spec=BaseChatModel)
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        strategy = DefaultRolloverStrategy(llm, backend)

        old_messages = [HumanMessage(content="old question")]
        lc_agent = mocker.Mock(spec=CompiledStateGraph)
        lc_agent.aget_state = AsyncMock(return_value=MagicMock(values={"messages": old_messages}))
        lc_agent.aupdate_state = AsyncMock()
        old_config: RunnableConfig = {"configurable": {"thread_id": old_thread_id}}

        # 試験実施
        new_thread_id, new_config = await strategy.invoke(
            lc_agent, "test-agent", old_thread_id, old_config
        )

        # 結果検証
        # 観点1
        assert new_thread_id != old_thread_id
        assert new_config.get("configurable", {}).get("thread_id") == new_thread_id
        lc_agent.aupdate_state.assert_not_called()

    async def test_summarize_and_offload_01(self):
        """旧 thread のメッセージを要約し、生ログを退避することを確認.

        観点1: backend へ history_path, 整形済みメッセージが退避され、read_file で参照できること
        観点2: llm の応答から要約メッセージ（HumanMessage）が生成されること
        """
        # 試験準備
        store = InMemoryStore()
        backend = StoreBackend(store=store, namespace=lambda _rt: ("test-agent", "filesystem"))
        llm = _FakeChatModel(messages=iter([AIMessage(content="summary text")]))
        messages = [HumanMessage(content="question"), AIMessage(content="answer")]

        # 試験実施
        result = await DefaultRolloverStrategy.summarize_and_offload(
            messages, "/history/thread.md", llm, backend
        )

        # 結果検証
        # 観点1
        history = await backend.aread("/history/thread.md")
        assert history.file_data is not None
        assert "question" in history.file_data["content"]
        assert "answer" in history.file_data["content"]
        # 観点2
        assert isinstance(result, HumanMessage)
        assert "summary text" in result.content


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
