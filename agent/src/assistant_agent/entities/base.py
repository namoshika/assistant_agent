from abc import abstractmethod

from langchain_core.documents import Document
from sqlalchemy import Engine, String, delete, insert
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.sql.elements import ColumnElement


class DocumentFields:
    """全エンティティ共通の基底クラス. backlink_filter を抽象メソッドとして定義."""

    document_id: Mapped[str] = mapped_column(String, primary_key=True, sort_order=0)
    document_metadata: Mapped[dict]  # mapped_column なし。sort_order=1 は具体クラスで指定
    content: Mapped[str] = mapped_column(String, nullable=False, sort_order=2)
    path: Mapped[str] = mapped_column(String, nullable=False, unique=True, sort_order=3)


class ObsidianFields(DocumentFields):
    @classmethod
    @abstractmethod
    def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
        """forward_links に document_id を含む行を絞り込む WHERE 句を返す."""
        raise NotImplementedError


class VaultUtils:
    """ドキュメントを DB に同期するクラス."""

    @staticmethod
    def sync(
        documents: list[Document],
        sa_engine: Engine,
        raw_entity: type[DocumentFields],
    ) -> None:
        """テーブルを引数 documents の内容で洗い替えする."""
        rows = [
            {
                "document_id": doc.id,
                "document_metadata": doc.metadata,
                "content": doc.page_content,
                "path": doc.metadata["path"],
            }
            for doc in documents
        ]
        assert rows is not None
        with Session(sa_engine) as session:
            session.execute(delete(raw_entity))
            session.execute(insert(raw_entity), rows)
            session.commit()
