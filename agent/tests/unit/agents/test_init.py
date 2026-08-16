import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from assistant_agent import agents
from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Receiver
from assistant_agent.utils.workflow import BroadcastPipe


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
        self.received: list[AgentInvocation] = []
        self._event = event

    def on_received(self, invocation: AgentInvocation) -> None:
        self.received.append(invocation)
        self._event.set()


@pytest.mark.parametrize("module_name", ["sample", "holo", "rune"])
async def test_get_agent_01(module_name: str) -> None:
    """各エージェントを get_agent() 経由で生成・実行できることを確認.

    観点1: 指定した module_name の Agent が応答を配信すること
    観点2: thread_id が module_name を prefix に持つこと
    """
    # 試験準備
    llm = _FakeChatModel(messages=iter([AIMessage(content="こんにちは。")]))
    checkpointer = InMemorySaver()
    agent = agents.get_agent(module_name, module_name, None, llm, {}, checkpointer, InMemoryStore())
    source = _DummyActiveEmitter()
    BroadcastPipe(source, [agent])
    received_event = asyncio.Event()
    received = _DummyReceiver(received_event)
    agent.receiver = received

    # 試験実施
    agent.start()
    source.emit(
        AgentInvocation(input={"messages": [HumanMessage(content="こんにちは。")]}, context={})
    )
    await asyncio.wait_for(received_event.wait(), timeout=5)

    # 結果検証
    # 観点1
    assert len(received.received) == 1
    # 観点2
    checkpoints = [c async for c in checkpointer.alist(None)]
    assert len(checkpoints) > 0
    thread_id = checkpoints[0].config["configurable"]["thread_id"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert thread_id.startswith(f"{module_name}:")


def test_get_agent_02() -> None:
    """get_agent() が存在しない module_name に対し ValueError を送出することを確認.

    観点1: 存在しない module_name を指定すると ValueError が送出されること
    """
    # 試験準備
    llm = _FakeChatModel(messages=iter([AIMessage(content="こんにちは。")]))

    # 試験実施、結果検証
    # 観点1
    with pytest.raises(ValueError, match="no-such-agent"):
        agents.get_agent(
            "no-such-agent", "no-such-agent", None, llm, {}, InMemorySaver(), InMemoryStore()
        )


async def test_get_agent_03() -> None:
    """get_agent() が agent_id を module_name と独立に扱うことを確認.

    観点1: module_name と異なる agent_id を指定すると、返る Agent の agent_id が
        指定した agent_id（module_name ではない）と一致すること
    """
    # 試験準備
    llm = _FakeChatModel(messages=iter([AIMessage(content="こんにちは。")]))
    checkpointer = InMemorySaver()

    # 試験実施
    agent = agents.get_agent("sample", "custom-id", None, llm, {}, checkpointer, InMemoryStore())
    source = _DummyActiveEmitter()
    BroadcastPipe(source, [agent])
    received_event = asyncio.Event()
    received = _DummyReceiver(received_event)
    agent.receiver = received
    agent.start()
    source.emit(
        AgentInvocation(input={"messages": [HumanMessage(content="こんにちは。")]}, context={})
    )
    await asyncio.wait_for(received_event.wait(), timeout=5)

    # 結果検証
    # 観点1
    checkpoints = [c async for c in checkpointer.alist(None)]
    assert len(checkpoints) > 0
    thread_id = checkpoints[0].config["configurable"]["thread_id"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert thread_id.startswith("custom-id:")


async def test_get_lc_agent_01() -> None:
    """get_lc_agent() の CompiledStateGraph が応答し、agent_id が backend へ伝播することを確認.

    観点1: sample モジュールを指定して得た CompiledStateGraph に ainvoke() で入力すると、
        実グラフの応答が返ること
    観点2: agent_id 固有の backend ディレクトリ（build_backend() が作成する agent_{agent_id}）が
        用意されること
    """
    # 試験準備
    llm = _FakeChatModel(messages=iter([AIMessage(content="こんにちは。")]))
    agent_id = f"test-{uuid.uuid4().hex}"
    profile_dir = Path(os.environ["AA_PROFILE_DIR"])

    # 試験実施
    lc_agent = agents.get_lc_agent("sample", agent_id, llm, None, InMemoryStore())
    result = await lc_agent.ainvoke({"messages": [HumanMessage(content="こんにちは。")]})

    # 結果検証
    # 観点1
    assert result["messages"][-1].content
    # 観点2
    assert (profile_dir / f"agent_{agent_id}").is_dir()


def test_get_lc_agent_02() -> None:
    """get_lc_agent() が存在しない module_name に対し ValueError を送出することを確認.

    観点1: 存在しない module_name を指定すると ValueError が送出されること
    """
    # 試験準備
    llm = _FakeChatModel(messages=iter([AIMessage(content="こんにちは。")]))

    # 試験実施、結果検証
    # 観点1
    with pytest.raises(ValueError, match="no-such-agent"):
        agents.get_lc_agent("no-such-agent", "no-such-agent", llm, None, InMemoryStore())


def test_build_backend_01() -> None:
    """build_backend() が agent_id 毎・skills 共有でファイル操作をルーティングすることを確認.

    観点1: "/" 配下への書き込みが agent_id 固有のディレクトリに保存されること
    観点2: "/skills/" 配下への書き込みが全エージェント共有のディレクトリに保存されること
    """
    # 試験準備
    agent_id = f"test-{uuid.uuid4().hex}"
    profile_dir = Path(os.environ["AA_PROFILE_DIR"])

    # 試験実施
    backend = agents.build_backend(agent_id)
    backend.write("/note.txt", "agent-scoped")
    backend.write("/skills/shared-note.txt", "shared")

    # 結果検証
    # 観点1
    assert (profile_dir / f"agent_{agent_id}" / "note.txt").read_text() == "agent-scoped"
    # 観点2
    assert (profile_dir / "skills" / "shared-note.txt").read_text() == "shared"
