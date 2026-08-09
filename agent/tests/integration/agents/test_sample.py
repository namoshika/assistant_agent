import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from assistant_agent import agents
from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver
from assistant_agent.utils.workflow import BroadcastPipe


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


@pytest.mark.integration
@pytest.mark.parametrize("module_name", ["sample", "holo", "rune"])
async def test_receive_01(module_name: str, llm: BaseChatModel) -> None:
    """各エージェントが実 LLM で入力を処理し、応答を配信できることを確認.

    観点1: BroadcastPipe で接続した発信元から新着を流すこと
    観点2: 各エージェントの応答が購読者へ配信されること
    """
    # 試験準備
    agent = agents.get_agent(
        module_name,
        module_name,
        None,
        llm,
        {},
        InMemorySaver(),
        InMemoryStore(),
    )
    source = _DummyActiveEmitter()
    BroadcastPipe(source, [agent])
    received_event = asyncio.Event()
    received = _DummyReceiver(received_event)
    agent.receiver = received

    try:
        # 試験実施
        invocation = AgentInvocation(
            input={"messages": [HumanMessage(content="こんにちは。自己紹介してください。")]},
            context={},
        )
        agent.start()
        source.emit(invocation)
        await asyncio.wait_for(received_event.wait(), timeout=30)

        # 結果検証
        # 観点1
        assert len(received.received) == 1
        # 観点2
        assert received.received[0]["input"]["messages"][-1].content
    finally:
        agent.stop()
