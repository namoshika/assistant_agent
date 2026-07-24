import abc

from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncEngine


class StoreConnector(abc.ABC):
    """SQLAlchemy AsyncEngine を生成するファクトリ抽象クラス."""

    @abc.abstractmethod
    def get_engine(self) -> AsyncEngine:
        """SQLAlchemy AsyncEngine を生成する."""
        raise NotImplementedError


class Receiver(abc.ABC):
    """メッセージを受信できる抽象クラス."""

    @abc.abstractmethod
    def on_received(self, msg: BaseMessage) -> None:
        """外部からメッセージを1件受け取る."""
        raise NotImplementedError


class Emitter(abc.ABC):
    """メッセージを発信できる抽象クラス（配信先の Receiver を1つ登録し、配信できる）."""

    def __init__(self):
        """配信先を未登録状態で初期化する."""
        self.receiver: Receiver | None = None

    def emit(self, msg: BaseMessage) -> None:
        """登録済みの配信先へ同期的に配信する（派生クラスはこれを使って送信する）."""
        if self.receiver is not None:
            self.receiver.on_received(msg)


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
