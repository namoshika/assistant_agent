import abc
from collections.abc import Callable

from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncEngine


class StoreConnector(abc.ABC):
    """SQLAlchemy AsyncEngine を生成するファクトリ抽象クラス."""

    @abc.abstractmethod
    def get_engine(self) -> AsyncEngine:
        """SQLAlchemy AsyncEngine を生成する."""
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
