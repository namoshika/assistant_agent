import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import langchain.agents
import pytest
from deepagents.backends.store import StoreBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver
from assistant_agent.utils.workflow import (
    THREAD_ROLLOVER_INTERVAL_DAYS,
    Agent,
    BroadcastPipe,
    DefaultRolloverStrategy,
)


class _DummyActiveEmitter(ActiveEmitter):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _DummyReceiver(Receiver):
    def __init__(self, event: asyncio.Event):
        self.received: list[AgentInvocation] = []
        self._event = event

    def on_received(self, invocation: AgentInvocation) -> None:
        self.received.append(invocation)
        self._event.set()


@pytest.fixture
def old_thread_id() -> str:
    """ロールオーバー対象と判定される（THREAD_ROLLOVER_INTERVAL_DAYS 以上前の）thread_id を返す."""
    old_time = datetime.now(UTC) - timedelta(days=THREAD_ROLLOVER_INTERVAL_DAYS + 1)
    old_ts_ms = int(old_time.timestamp() * 1000)
    base_uuid = uuid.uuid7()
    old_uuid_int = (base_uuid.int & ((1 << 80) - 1)) | (old_ts_ms << 80)
    return f"sample:{uuid.UUID(int=old_uuid_int)}"


class TestDefaultRolloverStrategy:
    @pytest.mark.integration
    async def test_invoke_01(self, llm: BaseChatModel, old_thread_id: str):
        """Agent に組み込んだ状態で、ターン処理中に thread_id がロールオーバーすることを確認.

        DefaultRolloverStrategy は Agent と結合して使うクラスのため、Agent 経由で確認する。

        観点1: 旧 thread の会話内容の要約が store へ書き込まれること（read_file で参照できること）
        観点2: ロールオーバー後の thread_id を含む config でグラフが呼ばれ、
            旧 thread の要約が新 thread の messages へ引き継がれた状態で応答が生成されること
        """
        # 試験準備
        checkpointer = InMemorySaver()
        store = InMemoryStore()
        lc_agent = langchain.agents.create_agent(llm, checkpointer=checkpointer)
        backend = StoreBackend(store=store, namespace=lambda _rt: ("sample", "filesystem"))
        rollover_strategy = DefaultRolloverStrategy(llm, backend)

        old_config: RunnableConfig = {"configurable": {"thread_id": old_thread_id}}
        await lc_agent.ainvoke(
            {"messages": [HumanMessage(content="こんにちは、私の好きな食べ物はりんごです。")]},
            config=old_config,
        )

        agent = Agent(
            lc_agent,
            context={},
            agent_id="sample",
            thread_id=old_thread_id,
            rollover_strategy=rollover_strategy,
        )
        source = _DummyActiveEmitter()
        BroadcastPipe(source, [agent])
        received_event = asyncio.Event()
        received = _DummyReceiver(received_event)
        agent.receiver = received

        # 試験実施
        agent.start()
        source.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content="私の好きな食べ物は何でしたか？")]},
                context={},
            )
        )
        await asyncio.wait_for(received_event.wait(), timeout=30)

        # 結果検証
        # 観点1
        history = await backend.aread(f"/conversation_history/{old_thread_id}.md")
        assert history.file_data is not None
        assert "りんご" in history.file_data["content"]
        # 観点2
        assert len(received.received) == 1
        assert "りんご" in received.received[0]["input"]["messages"][-1].text.lower()

    @pytest.mark.integration
    async def test_summarize_and_offload_01(self, llm: BaseChatModel) -> None:
        """実 LLM で旧 thread の会話を要約し、生ログを仮想ファイルシステムへ退避できることを確認.

        観点1: 生成された要約が HumanMessage として返り、要点（好きな食べ物）を含むこと
        観点2: 生ログが history_path へ退避され、元の会話内容を含むこと
        """
        # 試験準備
        store = InMemoryStore()
        backend = StoreBackend(store=store, namespace=lambda _rt: ("test-agent", "filesystem"))
        messages = [
            HumanMessage(content="こんにちは、私の好きな食べ物はりんごです。"),
            AIMessage(content="りんごがお好きなんですね、教えてくれてありがとうございます。"),
        ]
        history_path = "/conversation_history/test-agent:test-thread.md"

        # 試験実施
        summary_message = await DefaultRolloverStrategy.summarize_and_offload(
            messages, history_path, llm, backend
        )

        # 結果検証
        # 観点1
        assert "apple" in summary_message.text.lower()
        # 観点2
        history = await backend.aread(history_path)
        assert history.file_data is not None
        assert "りんご" in history.file_data["content"]
