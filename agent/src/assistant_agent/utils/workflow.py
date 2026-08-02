import asyncio
import logging
import uuid
from typing import Any, cast

import mlflow
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import SpanType

from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation, Emitter, Receiver
from assistant_agent.utils.context import CommonContext

TIMEOUT_SECONDS_DEFAULT = 300
SYNC_REQUEST_MESSAGE_TMPL = """\
# {channel_name}
## Guideline
イベントを受信しました。このイベントに対しては行動後に理由ではなく、応答自体を出力してください。

## Received Prompt
{content}
"""


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
        self._queue: asyncio.Queue[AgentInvocation] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._logger = logging.getLogger(__name__)

    def on_received(self, invocation: AgentInvocation) -> None:
        """BroadcastPipe から渡された AgentInvocation をキューへ積む."""
        self._queue.put_nowait(invocation)

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
                invocation = await self._queue.get()
                received_context = invocation["context"]
                timeout_seconds = received_context.get("timeout_seconds", TIMEOUT_SECONDS_DEFAULT)
                try:
                    merged_invocation = cast(
                        AgentInvocation,
                        invocation | {"context": self._context | received_context},
                    )
                    span.set_inputs(invocation["input"]["messages"][-1].content)
                    self._logger.info(self._format_log_message("Consume a message"))
                    result = await asyncio.wait_for(
                        self._agent.ainvoke(**merged_invocation, config=config, version="v2"),
                        timeout=timeout_seconds,
                    )
                    msg_out = result.value["messages"][-1]
                    span.set_outputs(msg_out)
                    mlflow.update_current_trace(
                        metadata={
                            "mlflow.trace.user": "123",
                            "mlflow.trace.session": self._thread_id,
                        }
                    )
                except TimeoutError:
                    txt = self._format_log_message(f"ainvoke() timed out after {timeout_seconds}s")
                    self._logger.error(txt)
                    msg_out = AIMessage(content=txt)
                except Exception:
                    txt = self._format_log_message("Exception during execution")
                    self._logger.exception(txt)
                    msg_out = AIMessage(content=txt)
                self.emit(AgentInvocation(input={"messages": [msg_out]}, context=received_context))

    def _format_log_message(self, message: str) -> str:
        return f"{message} (thread_id: {self._thread_id}, queue: {self._queue.qsize()},  trace_id: {mlflow.get_last_active_trace_id()})"  # noqa: E501


class SyncRequestChannel(ActiveEmitter, Receiver):
    """Agent のキューへメッセージを投入し、対応する応答を同期的に待ち受ける汎用チャンネル."""

    def __init__(self) -> None:
        """SyncRequestChannel を構成する."""
        super().__init__()
        self._pending: dict[str, asyncio.Future[BaseMessage]] = {}

    def start(self) -> None:
        """待受状態を開始する（内部状態のみのため何もしない）."""

    def stop(self) -> None:
        """未解決の Future を全てキャンセルする."""
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    def on_received(self, invocation: AgentInvocation) -> None:
        """Agent からの応答を、context.request_id に対応する Future へ引き渡す."""
        request_id = invocation["context"].get("request_id")
        if request_id is None or request_id not in self._pending:
            return
        fut = self._pending.pop(request_id)
        if not fut.done():
            fut.set_result(invocation["input"]["messages"][-1])

    async def emit_and_wait(
        self, content: str, channel_name: str, timeout_seconds: int = 300
    ) -> BaseMessage:
        """引数 content を Agent へ emit し、対応する応答を待って返す.

        timeout_seconds 秒以内に応答が得られない場合は TimeoutError を送出する。
        """
        request_id = str(uuid.uuid7())
        fut: asyncio.Future[BaseMessage] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        prompt = SYNC_REQUEST_MESSAGE_TMPL.format(
            request_id=request_id, content=content, channel_name=channel_name
        )
        self.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content=prompt)]},
                context={"request_id": request_id},
            )
        )
        try:
            return await asyncio.wait_for(fut, timeout=timeout_seconds)
        finally:
            self._pending.pop(request_id, None)


class BroadcastPipe(Receiver):
    """Emitter 1つと Receiver 複数を1対nで接続する（1対1の接続にも使う）."""

    def __init__(self, src: Emitter, dst: list[Receiver]):
        """src（発信元）から dst（受信先の一覧）への配線を構成する."""
        src.receiver = self
        self._dst = dst

    def on_received(self, invocation: AgentInvocation) -> None:
        """発信元から受け取ったメッセージを dst の全要素へ配信する."""
        for d in self._dst:
            d.on_received(invocation)


class MergePipe(Receiver):
    """Emitter 複数と Receiver 1つをn対1で接続する."""

    def __init__(self, src: list[Emitter], dst: Receiver):
        """src（発信元の一覧）から dst（受信先）への配線を構成する."""
        self._dst = dst
        for s in src:
            s.receiver = self

    def on_received(self, invocation: AgentInvocation) -> None:
        """いずれかの発信元から受け取ったメッセージを dst へ中継する."""
        self._dst.on_received(invocation)


class LogWriter(Receiver):
    """受け取ったメッセージを logging 経由で記録する（ハンドラ設定は呼び出し側が行う）."""

    def __init__(self) -> None:
        """LogWriter を初期化."""
        super().__init__()
        self._logger = logging.getLogger(__name__)

    def on_received(self, invocation: AgentInvocation) -> None:
        """メッセージ本文を INFO レベルで記録する."""
        content = invocation["input"]["messages"][-1].content
        self._logger.debug(f"Called agent (trace_id: {mlflow.get_active_trace_id()}):\n{content}")
