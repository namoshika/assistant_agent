import abc
from typing import Any, Required, TypedDict

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncEngine

from assistant_agent.utils.context import CommonContext


class StoreConnector(abc.ABC):
    """SQLAlchemy AsyncEngine を生成するファクトリ抽象クラス."""

    @abc.abstractmethod
    def get_engine(self) -> AsyncEngine:
        """SQLAlchemy AsyncEngine を生成する."""
        raise NotImplementedError


class RolloverStrategy(abc.ABC):
    """thread_id の世代交代要否を判定し、必要なら実施する戦略の抽象クラス."""

    @abc.abstractmethod
    async def invoke(
        self,
        lc_agent: CompiledStateGraph[Any, Any, Any, Any],
        agent_id: str,
        thread_id: str,
        config: RunnableConfig,
    ) -> tuple[str, RunnableConfig]:
        """現在の thread_id, config を判定し、必要なら新しい thread_id, config を返す."""
        # 呼び出しのたびに現在の thread_id, config を受け取り、
        # ロールオーバーしない場合は入力をそのまま、する場合は新しい thread_id, config を返す。
        # 要約の引き継ぎ（新 thread の checkpoint への差し込み等）が必要な場合はここで完結させる。
        # 要約生成・退避に使う llm, backend 等は、実装クラスのコンストラクタで受け取り保持する
        # （Agent はそれらを知らない）。
        raise NotImplementedError


class Receiver(abc.ABC):
    """メッセージを受信できる抽象クラス."""

    @abc.abstractmethod
    def on_received(self, invocation: AgentInvocation) -> None:
        """外部から AgentInvocation を1件受け取る."""
        raise NotImplementedError


class Emitter(abc.ABC):
    """メッセージを発信できる抽象クラス（配信先の Receiver を1つ登録し、配信できる）."""

    def __init__(self):
        """配信先を未登録状態で初期化する."""
        self.receiver: Receiver | None = None

    def emit(self, invocation: AgentInvocation) -> None:
        """登録済みの配信先へ同期的に配信する（派生クラスはこれを使って送信する）."""
        if self.receiver is not None:
            self.receiver.on_received(invocation)


class ActiveEmitter(Emitter):
    """能動的に動作できる Emitter（起動・停止を持つ）."""

    @abc.abstractmethod
    def start(self) -> None:
        """動作を開始する."""
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        """動作を終了する."""
        raise NotImplementedError


class AgentInvocation(TypedDict, total=False):
    """Agent.ainvoke() へ `**invocation` でそのまま展開して渡せるキーワード引数の集合.

    input, context 以外は stream_mode 等 ainvoke() が受け付ける任意のキーワード引数を、
    既存の構築箇所を変更せず追加できる拡張の余地として残す。
    """

    input: Required[dict[str, Any]]
    context: Required[CommonContext]
