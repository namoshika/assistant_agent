import asyncio
import logging
import uuid
from typing import Any

import mlflow
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import SpanType

from assistant_agent.utils.absclass import ActiveEmitter, Emitter, Receiver
from assistant_agent.utils.context import CommonContext


class Agent(ActiveEmitter, Receiver):
    """BroadcastPipe から入力を受け取り応答を発信するエージェント."""

    def __init__(
        self,
        lc_agent: CompiledStateGraph[Any, Any, Any, Any],
        context: CommonContext,
        thread_id: str | None = None,
    ):
        """Agentを構成する."""
        super().__init__()
        self._agent = lc_agent
        self._thread_id = thread_id or str(uuid.uuid7())
        self._context = context
        self._queue: asyncio.Queue[BaseMessage] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._logger = logging.getLogger(__name__)

    def on_received(self, msg: BaseMessage) -> None:
        """BroadcastPipe から渡されたメッセージをキューへ積む."""
        self._queue.put_nowait(msg)

    def start(self) -> None:
        """新着の処理を開始する（起動済みなら何もしない）."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._consume())

    def stop(self) -> None:
        """新着の処理を停止する."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _consume(self) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}
        while True:
            with mlflow.start_span(span_type=SpanType.CHAT_MODEL) as span:
                try:
                    msg_in = await self._queue.get()
                    span.set_inputs(msg_in.content)
                    self._logger.info(self._format_log_message("Consume a message"))
                    result = await self._agent.ainvoke(
                        {"messages": [msg_in]}, config=config, context=self._context, version="v2"
                    )
                    msg_out = result.value["messages"][-1]
                    span.set_outputs(msg_out)
                    mlflow.update_current_trace(
                        metadata={
                            "mlflow.trace.user": "123",
                            "mlflow.trace.session": self._thread_id,
                        }
                    )
                    self.emit(msg_out)
                except Exception:
                    self._logger.exception(self._format_log_message("Exception during execution"))

    def _format_log_message(self, message: str) -> str:
        return f"{message} (thread_id: {self._thread_id}, queue: {self._queue.qsize()},  trace_id: {mlflow.get_last_active_trace_id()})"  # noqa: E501


class BroadcastPipe(Receiver):
    """Emitter 1つと Receiver 複数を1対nで接続する（1対1の接続にも使う）."""

    def __init__(self, src: Emitter, dst: list[Receiver]):
        """src（発信元）から dst（受信先の一覧）への配線を構成する."""
        src.receiver = self
        self._dst = dst

    def on_received(self, msg: BaseMessage) -> None:
        """発信元から受け取ったメッセージを dst の全要素へ配信する."""
        for d in self._dst:
            d.on_received(msg)


class MergePipe(Receiver):
    """Emitter 複数と Receiver 1つをn対1で接続する."""

    def __init__(self, src: list[Emitter], dst: Receiver):
        """src（発信元の一覧）から dst（受信先）への配線を構成する."""
        self._dst = dst
        for s in src:
            s.receiver = self

    def on_received(self, msg: BaseMessage) -> None:
        """いずれかの発信元から受け取ったメッセージを dst へ中継する."""
        self._dst.on_received(msg)


class LogWriter(Receiver):
    """受け取ったメッセージを logging 経由で記録する（ハンドラ設定は呼び出し側が行う）."""

    def __init__(self) -> None:
        """LogWriter を初期化."""
        super().__init__()
        self._logger = logging.getLogger(__name__)

    def on_received(self, msg: BaseMessage) -> None:
        """メッセージ本文を INFO レベルで記録する."""
        self._logger.debug(
            f"Called agent (trace_id: {mlflow.get_active_trace_id()}):\n{msg.content}"
        )
