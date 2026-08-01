from typing import Any

from langchain_core.messages import HumanMessage

from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver


class _DummyActiveEmitter(ActiveEmitter):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _DummyReceiver(Receiver):
    def __init__(self):
        self.received: list[Any] = []

    def on_received(self, invocation: AgentInvocation) -> None:
        self.received.append(invocation)


class TestEmitter:
    def test_emit_01(self):
        """receiver/emit の配信挙動を確認.

        観点1: 登録した配信先の receive が emit で呼ばれる
        観点2: 後から receiver を代入し直すと配信先が上書きされ、新しい方のみ呼ばれる
        """
        # 試験準備
        emitter = _DummyActiveEmitter()
        dst1 = _DummyReceiver()
        dst2 = _DummyReceiver()
        invocation: AgentInvocation = {"input": {"messages": [HumanMessage(content="hello")]}}

        # 試験実施
        emitter.receiver = dst1
        emitter.receiver = dst2
        emitter.emit(invocation)

        # 結果検証
        # 観点1
        assert dst2.received == [invocation]
        # 観点2
        assert dst1.received == []
