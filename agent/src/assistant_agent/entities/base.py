from abc import abstractmethod

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement


class ObsidianVaultEntity:
    """全エンティティ共通の基底クラス. backlink_filter を抽象メソッドとして定義."""

    document_id: Mapped[str] = mapped_column(String, primary_key=True, sort_order=0)
    document_metadata: Mapped[dict]  # mapped_column なし。sort_order=1 は具体クラスで指定
    content: Mapped[str] = mapped_column(String, nullable=False, sort_order=2)
    path: Mapped[str] = mapped_column(String, nullable=False, unique=True, sort_order=3)

    @classmethod
    @abstractmethod
    def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
        """forward_links に document_id を含む行を絞り込む WHERE 句を返す."""
        raise NotImplementedError
