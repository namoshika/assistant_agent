import abc
import asyncio
import uuid
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, GraphOutput

from assistant_agent.utils.absclass import Channel


class BaseAgent(Channel):
    """Channel を購読して応答するエージェントの抽象クラス."""

    def __init__(
        self,
        channel: Channel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        thread_id: str | None = None,
    ):
        """channel（実行契機）・checkpointer（会話履歴）の組み合わせを構成する."""
        super().__init__()
        self._channel = channel
        self._checkpointer = checkpointer
        self._thread_id = thread_id or str(uuid.uuid7())
        self._queue: asyncio.Queue[BaseMessage] = asyncio.Queue()
        if channel is not None:
            channel.subscribe(self._queue.put_nowait)
        self._task: asyncio.Task[None] | None = None
        self._agent = self._build_agent()

    @property
    def lc_agent(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """コアの LangGraph のエージェント."""
        return self._agent

    @abc.abstractmethod
    def _build_agent(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """具象エージェントのグラフを構築する（Factory Method）."""
        raise NotImplementedError

    async def invoke(
        self,
        value: dict[str, Any] | Command | None,
        config: RunnableConfig | None = None,
        context: Any | None = None,
    ) -> GraphOutput[Any]:
        """単発実行として1回分の入力を渡し、応答を受け取る（必要に応じて具象クラス側でオーバーライドする）."""
        return await self._agent.ainvoke(value, config=config, context=context, version="v2")

    async def start(self) -> None:
        """継続購読を開始する（channel なし利用時は no-op）."""
        if self._channel is None:
            return
        self._task = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        """継続購読を停止する（channel なし利用時は no-op）."""
        if self._channel is None:
            return
        if self._task is not None:
            self._task.cancel()

    async def _consume(self) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}
        while True:
            msg = await self._queue.get()  # 新着まで待機（ポーリングしない）
            result = await self.invoke({"messages": [msg]}, config=config)
            self.publish(result.value["messages"][-1])
