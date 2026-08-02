import asyncio
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from assistant_agent.agents import sample
from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver
from assistant_agent.utils.workflow import Agent, BroadcastPipe


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
async def test_receive_01():
    """実 LLM で新着に応答し、BroadcastPipe 経由で配信されるか確認.

    観点1: BroadcastPipe で接続した発信元から新着を流すと、
        実グラフの応答が Agent の購読者へ配信されること
    """
    # 試験準備
    lc_agent = sample.build_lc_agent()
    agent = Agent(lc_agent, context={})
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
    await asyncio.wait_for(received_event.wait(), timeout=30)

    # 結果検証
    # 観点1
    assert len(received.received) == 1
    assert received.received[0]["input"]["messages"][-1].content
