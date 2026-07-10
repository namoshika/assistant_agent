import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import langchain.agents
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from assistant_agent.agents.base import BaseAgent
from assistant_agent.utils.absclass import Channel


class _DummyChannel(Channel):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> "_FakeChatModel":
        return self


class _TestableAgent(BaseAgent):
    def __init__(self, mock_graph: CompiledStateGraph, **kwargs: Any):
        self._mock_graph = mock_graph
        super().__init__(**kwargs)

    def _build_agent(self) -> CompiledStateGraph:
        return self._mock_graph


class TestBaseAgent:
    async def test_invoke_01(self):
        """invoke() の戻り値と checkpointer による履歴継続の有無を確認.

        観点1: invoke(value) が実グラフの GraphOutput をそのまま返す
        観点2: checkpointer ありでは2回目の invoke で履歴が引き継がれ、
            checkpointer なしでは引き継がれない
        """
        # 試験準備
        graph_cp_on = langchain.agents.create_agent(
            model=_FakeChatModel(
                messages=iter([AIMessage(content="reply1"), AIMessage(content="reply2")])
            ),
            tools=[],
            system_prompt="test",
            checkpointer=InMemorySaver(),
        )
        agent_cp_on = _TestableAgent(graph_cp_on)
        graph_cp_off = langchain.agents.create_agent(
            model=_FakeChatModel(
                messages=iter([AIMessage(content="reply1"), AIMessage(content="reply2")])
            ),
            tools=[],
            system_prompt="test",
            checkpointer=None,
        )
        agent_cp_off = _TestableAgent(graph_cp_off)
        config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}

        # 試験実施
        res_1st_cp_on = await agent_cp_on.invoke(
            {"messages": [HumanMessage(content="first")]}, config
        )
        res_2nd_cp_on = await agent_cp_on.invoke(
            {"messages": [HumanMessage(content="second")]}, config
        )
        res_1st_cp_off = await agent_cp_off.invoke(
            {"messages": [HumanMessage(content="first")]}, config
        )
        res_2nd_cp_off = await agent_cp_off.invoke(
            {"messages": [HumanMessage(content="second")]}, config
        )

        # 結果検証
        # 観点1
        assert isinstance(res_1st_cp_on, GraphOutput)
        assert res_1st_cp_on.value["messages"][-1].content == "reply1"
        assert res_1st_cp_off.value["messages"][-1].content == "reply1"
        # 観点2: checkpointer ありは履歴が蓄積され4件、なしは毎回2件
        assert len(res_2nd_cp_on.value["messages"]) == 4
        assert len(res_2nd_cp_off.value["messages"]) == 2

    async def test_start_01(self):
        """start()/stop() による Channel 購読のライフサイクルを確認.

        観点1: start() 後に publish() で新着を流すと ainvoke が呼ばれ、
            Agent 自身の購読者へ応答が配信される（Agent が Channel として連鎖できる）
        観点2: 同一 channel を購読する2つの Agent が独立して応答を配信する
        観点3: stop() 後は publish() しても ainvoke が呼ばれず配信されない
        """
        # 試験準備
        channel = _DummyChannel()
        mock_graph1 = MagicMock(spec=CompiledStateGraph)
        mock_graph1.ainvoke = AsyncMock(
            return_value=GraphOutput(value={"messages": [AIMessage(content="reply1")]})
        )
        mock_graph2 = MagicMock(spec=CompiledStateGraph)
        mock_graph2.ainvoke = AsyncMock(
            return_value=GraphOutput(value={"messages": [AIMessage(content="reply2")]})
        )
        agent1 = _TestableAgent(mock_graph1, channel=channel)
        agent2 = _TestableAgent(mock_graph2, channel=channel)
        received1 = MagicMock()
        received2 = MagicMock()
        agent1.subscribe(received1)
        agent2.subscribe(received2)

        # 試験実施
        await agent1.start()
        await agent2.start()
        channel.publish(HumanMessage(content="hi"))
        await asyncio.sleep(0.05)

        # 結果検証
        # 観点1
        mock_graph1.ainvoke.assert_called_once()
        received1.assert_called_once()
        # 観点2
        mock_graph2.ainvoke.assert_called_once()
        received2.assert_called_once()

        # 試験実施: stop 後は配信されない
        await agent1.stop()
        await agent2.stop()
        channel.publish(HumanMessage(content="after stop"))
        await asyncio.sleep(0.05)

        # 結果検証
        # 観点3
        mock_graph1.ainvoke.assert_called_once()
        received1.assert_called_once()

    async def test_start_02(self):
        """channel=None の場合、start()/stop() が no-op であることを確認.

        観点1: channel=None の場合 start()/stop() が no-op で例外を出さない
        """
        # 試験準備
        agent = _TestableAgent(MagicMock(spec=CompiledStateGraph), channel=None)

        # 試験実施・結果検証
        # 観点1: 例外が出ずここまで到達すれば成功
        await agent.start()
        await agent.stop()
