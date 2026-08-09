import asyncio
from typing import Any

from deepagents.backends.store import StoreBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from assistant_agent.agents import sample
from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver
from assistant_agent.utils.workflow import Agent, BroadcastPipe, DefaultRolloverStrategy


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> _FakeChatModel:
        return self


class _DummyActiveEmitter(ActiveEmitter):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _DummyReceiver(Receiver):
    def __init__(self, event: asyncio.Event):
        self.received: list[Any] = []
        self._event = event

    def on_received(self, invocation: AgentInvocation) -> None:
        self.received.append(invocation)
        self._event.set()


async def test_receive_01() -> None:
    """build_lc_agent() で構築したグラフを Agent 化し、新着に応答して emit することを確認.

    観点1: BroadcastPipe で接続した発信元から新着を流すと、
        グラフの応答が Agent の購読者へ配信されること
    """
    # 試験準備
    llm = _FakeChatModel(messages=iter([AIMessage(content="こんにちは、assistant_agent_1です。")]))
    store = InMemoryStore()
    lc_agent = sample.build_lc_agent(InMemorySaver(), store, llm, "sample")
    backend = StoreBackend(
        store=store,
        namespace=lambda _rt: ("sample", "filesystem"),
    )
    agent = Agent(
        lc_agent,
        context={},
        agent_id="sample",
        thread_id=None,
        rollover_strategy=DefaultRolloverStrategy(llm, backend),
    )
    source = _DummyActiveEmitter()
    BroadcastPipe(source, [agent])
    received_event = asyncio.Event()
    received = _DummyReceiver(received_event)
    agent.receiver = received

    # 試験実施
    agent.start()
    invocation = AgentInvocation(
        input={"messages": [HumanMessage(content="こんにちは。自己紹介してください。")]},
        context={},
    )
    source.emit(invocation)
    await asyncio.wait_for(received_event.wait(), timeout=5)

    # 結果検証
    # 観点1
    assert len(received.received) == 1
    assert (
        received.received[0]["input"]["messages"][-1].content
        == "こんにちは、assistant_agent_1です。"
    )
