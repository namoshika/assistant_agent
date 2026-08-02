import abc
from typing import Any, Required, TypedDict

from sqlalchemy.ext.asyncio import AsyncEngine

from assistant_agent.utils.context import CommonContext


class StoreConnector(abc.ABC):
    """SQLAlchemy AsyncEngine を生成するファクトリ抽象クラス."""

    @abc.abstractmethod
    def get_engine(self) -> AsyncEngine:
        """SQLAlchemy AsyncEngine を生成する."""
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
