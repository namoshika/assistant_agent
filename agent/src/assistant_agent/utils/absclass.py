import abc
import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, GraphOutput
from sqlalchemy import Engine


class StoreConnector(abc.ABC):
    """VectorStore / SQLAlchemy Engine を生成するファクトリ抽象クラス."""

    @abc.abstractmethod
    def get_engine(self) -> Engine:
        """SQLAlchemy Engine を生成する."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_vector_store(self, entity: type, embedding: Embeddings) -> VectorStore:
        """VectorStore を生成する."""
        raise NotImplementedError


class Channel(abc.ABC):
    """外部メッセージ源を購読者へ配信する抽象クラス."""

    def __init__(self):
        """購読者集合を初期化する."""
        # set: 同じコールバックの重複登録を防ぐ
        self._subscribers: set[Callable[[BaseMessage], None]] = set()

    def subscribe(self, callback: Callable[[BaseMessage], None]) -> None:
        """購読者を登録する（複数の購読者から呼ばれる想定）."""
        self._subscribers.add(callback)

    def publish(self, msg: BaseMessage) -> None:
        """登録済みの購読者全員へ同期的に配信する（派生クラスはこれを使って送信する）."""
        for cb in self._subscribers:
            cb(msg)  # 同期呼び出し。重い処理はしない

    @abc.abstractmethod
    async def start(self) -> None:
        """外部接続を開始する."""
        raise NotImplementedError

    @abc.abstractmethod
    async def stop(self) -> None:
        """外部接続を終了する."""
        raise NotImplementedError


class Agent(Channel):
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
        # checkpointer が None の場合でも常に確定する（LangGraph 側が無視するだけで実害がないため）
        self._thread_id = thread_id or str(uuid.uuid7())
        # channel なし利用時も未使用のまま保持する
        self._queue: asyncio.Queue[BaseMessage] = asyncio.Queue()
        if channel is not None:
            channel.subscribe(self._queue.put_nowait)
        self._task: asyncio.Task[None] | None = None
        # build_agent() は checkpointer 利用時に self._checkpointer を参照する場合がある
        self._agent = self.build_agent()

    @abc.abstractmethod
    def build_agent(self) -> CompiledStateGraph:
        """具象エージェントのグラフを構築する（Factory Method）."""
        raise NotImplementedError

    async def invoke(
        self, value: dict[str, Any] | Command | None, config: RunnableConfig | None = None
    ) -> GraphOutput[Any]:
        """単発実行として1回分の入力を渡し、応答を受け取る（必要に応じて具象クラス側でオーバーライドする）."""
        return await self._agent.ainvoke(value, config=config, version="v2")

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
