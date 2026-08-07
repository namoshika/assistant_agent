import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import mlflow
from deepagents.backends.store import StoreBackend
from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import SpanType
from mlflow.langchain.utils.chat import convert_lc_message_to_chat_message

from assistant_agent.utils.absclass import (
    ActiveEmitter,
    AgentInvocation,
    Emitter,
    Receiver,
    RolloverStrategy,
)
from assistant_agent.utils.context import CommonContext

TIMEOUT_SECONDS_DEFAULT = 300
SUMMARY_TIMEOUT_SECONDS = 300  # ロールオーバー時の要約生成のタイムアウト
THREAD_ROLLOVER_INTERVAL_DAYS = 3  # thread_id の世代交代間隔（日数）
# MLflow UI の Chat タブは mlflow.chat.messages 属性を読んで表示する
# (このバージョンの mlflow には SpanAttributeKey.CHAT_MESSAGES 定数が無いため直接指定する)
CHAT_MESSAGES_ATTR_KEY = "mlflow.chat.messages"

SYNC_REQUEST_MESSAGE_TMPL = """\
# {channel_name}
## Guideline
イベントを受信しました。このイベントに対しては行動後に理由ではなく、応答自体を出力してください。

## Received Prompt
{content}
"""
SUMMARY_HANDOFF_TMPL = """\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {history_path} should you need to
refer back to it for details.

<summary>
{summary}
</summary>
"""


class Agent(ActiveEmitter, Receiver):
    """BroadcastPipe から入力を受け取り応答を発信するエージェント."""

    def __init__(
        self,
        lc_agent: CompiledStateGraph[Any, Any, Any, Any],
        context: CommonContext,
        agent_id: str,
        thread_id: str | None,
        rollover_strategy: RolloverStrategy,
    ):
        """Agentを構成する."""
        super().__init__()
        self._agent = lc_agent
        self._agent_id = agent_id
        self._thread_id = thread_id or self._new_thread_id()
        self._context = context | {"agent_id": agent_id}
        self._rollover_strategy = rollover_strategy
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
            invocation = await self._queue.get()
            received_context = invocation["context"]
            agent_id = received_context.get("agent_id")
            if agent_id is not None and agent_id != self._agent_id:
                continue
            with mlflow.start_span("Agent", span_type=SpanType.CHAT_MODEL) as span:
                timeout_seconds = received_context.get("timeout_seconds", TIMEOUT_SECONDS_DEFAULT)
                try:
                    # スレッドをロールオーバー (戦略の判断に応じて実施される)
                    self._thread_id, config = await self._rollover_strategy.invoke(
                        self._agent, self._agent_id, self._thread_id, config
                    )
                    mlflow.update_current_trace(
                        metadata={
                            "mlflow.trace.user": "123",
                            "mlflow.trace.session": self._thread_id,
                        }
                    )

                    # エージェントを呼び出しオブジェクトを準備
                    merged_invocation = cast(
                        AgentInvocation,
                        invocation | {"context": received_context | self._context},
                    )
                    msg_in = invocation["input"]["messages"][-1]
                    chat_msg_in = convert_lc_message_to_chat_message(msg_in).model_dump()
                    span.set_inputs({"messages": [chat_msg_in]})
                    self._logger.info(self._format_log_message("Consume a message"))

                    # エージェントを呼び出し
                    result = await asyncio.wait_for(
                        self._agent.ainvoke(**merged_invocation, config=config, version="v2"),
                        timeout=timeout_seconds,
                    )
                    msg_out = result.value["messages"][-1]
                    # Responses API に差し込まれる type=reasoning ブロックは、mlflow の
                    # ChatMessage（text/image_url/input_audio のみ許容）が検証エラーとするため除去
                    chat_msg_out = convert_lc_message_to_chat_message(
                        self._drop_reasoning_content(msg_out)
                    ).model_dump()
                    span.set_outputs({"messages": [chat_msg_out]})
                    span.set_attribute(CHAT_MESSAGES_ATTR_KEY, [chat_msg_in, chat_msg_out])
                except TimeoutError:
                    txt = self._format_log_message(f"ainvoke() timed out after {timeout_seconds}s")
                    self._logger.error(txt)
                    msg_out = AIMessage(content=txt)
                except Exception:
                    txt = self._format_log_message("Exception during execution")
                    self._logger.exception(txt)
                    msg_out = AIMessage(content=txt)

            # 後続へ送信
            self.emit(AgentInvocation(input={"messages": [msg_out]}, context=received_context))

    def _format_log_message(self, message: str) -> str:
        return f"{message} (thread_id: {self._thread_id}, queue: {self._queue.qsize()},  trace_id: {mlflow.get_last_active_trace_id()})"  # noqa: E501

    def _new_thread_id(self) -> str:
        return f"{self._agent_id}:{uuid.uuid7()}"

    @staticmethod
    def _drop_reasoning_content(message: AIMessage) -> AIMessage:
        """Content 内の reasoning ブロックを取り除いた複製を返す."""
        if not isinstance(message.content, list):
            return message
        filtered = [
            b for b in message.content if not (isinstance(b, dict) and b.get("type") == "reasoning")
        ]
        return message.model_copy(update={"content": filtered})


class DefaultRolloverStrategy(RolloverStrategy):
    """thread_id の経過日数のみで判定し、無条件にロールオーバーする既定の戦略.

    要約は新 thread の checkpoint（messages）へ aupdate_state() で直接差し込む。
    要約・退避に失敗しても世代交代自体は継続する（要約が無くても新 thread への
    移行は成立するため）。要約生成・生ログ退避に使う llm・backend はコンストラクタで受け取る。
    """

    def __init__(self, llm: BaseChatModel, backend: StoreBackend) -> None:
        """DefaultRolloverStrategy を構成する."""
        self._llm = llm
        self._backend = backend
        self._logger = logging.getLogger(__name__)

    async def invoke(
        self,
        lc_agent: CompiledStateGraph[Any, Any, Any, Any],
        agent_id: str,
        thread_id: str,
        config: RunnableConfig,
    ) -> tuple[str, RunnableConfig]:
        """現在の thread_id・config を判定し、必要なら新しい thread_id・config を返す."""
        if not self.should_rollover(thread_id):
            return thread_id, config

        new_thread_id = f"{agent_id}:{uuid.uuid7()}"
        new_config: RunnableConfig = {"configurable": {"thread_id": new_thread_id}}

        snapshot = await lc_agent.aget_state(config)
        messages = snapshot.values.get("messages", [])
        if len(messages) > 0:
            try:
                summary_message = await self.summarize_and_offload(
                    messages, f"/conversation_history/{thread_id}.md", self._llm, self._backend
                )
                await lc_agent.aupdate_state(new_config, {"messages": [summary_message]})
            except Exception:
                self._logger.exception(f"Failed to summarize/offload (thread_id: {thread_id})")

        self._logger.info(f"Rolled over thread_id: {thread_id} -> {new_thread_id}")
        return new_thread_id, new_config

    @staticmethod
    def should_rollover(thread_id: str) -> bool:
        """thread_id に含まれる uuid7 部分から生成時刻を取り出し、経過日数で判定する.

        想定外の形式（移行前に払い出された prefix 無しの thread_id 等）は
        世代交代の対象として True を返し、新しい形式へ移行させる。
        """
        _, _, uuid_part = thread_id.partition(":")
        try:
            created_at = datetime.fromtimestamp(uuid.UUID(uuid_part).time / 1000, tz=UTC)
        except ValueError:
            return True
        return datetime.now(UTC) - created_at >= timedelta(days=THREAD_ROLLOVER_INTERVAL_DAYS)

    @staticmethod
    async def summarize_and_offload(
        messages: list[AnyMessage],
        history_path: str,
        llm: BaseChatModel,
        backend: StoreBackend,
    ) -> HumanMessage:
        """旧 thread の全メッセージを要約し、生ログを仮想ファイルシステムへ退避する."""
        formatted = get_buffer_string(messages, format="xml")
        await backend.awrite(history_path, formatted)  # 要約より先に退避
        response = await asyncio.wait_for(
            llm.ainvoke(DEFAULT_SUMMARY_PROMPT.format(messages=formatted)),
            timeout=SUMMARY_TIMEOUT_SECONDS,
        )
        content = SUMMARY_HANDOFF_TMPL.format(
            history_path=history_path, summary=response.text.strip()
        )
        return HumanMessage(content=content, additional_kwargs={"lc_source": "summarization"})


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
