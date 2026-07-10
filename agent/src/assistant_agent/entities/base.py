import hashlib
from abc import abstractmethod
from collections.abc import Sequence

from langchain_core.documents import Document
from sqlalchemy import JSON, Row, String, delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement


class DocumentFields:
    """ドキュメントテーブルの共通カラム."""

    document_id: Mapped[str] = mapped_column(String, primary_key=True, sort_order=0)
    document_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=False, sort_order=1)
    document_content_hash: Mapped[str] = mapped_column(String, nullable=False, sort_order=2)
    content: Mapped[str] = mapped_column(String, nullable=False, sort_order=3)
    file_path: Mapped[str] = mapped_column(String, nullable=False, unique=True, sort_order=4)


class ChunkFields:
    """チャンクテーブルの共通カラム（postgres 固有型は派生クラスでオーバーライドする）."""

    langchain_id: Mapped[str] = mapped_column(String, primary_key=True, sort_order=0)
    langchain_metadata: Mapped[dict | None] = mapped_column(JSON, sort_order=1)
    document_id: Mapped[str] = mapped_column(String, nullable=False, sort_order=2)
    document_content_hash: Mapped[str] = mapped_column(String, nullable=False, sort_order=3)
    content: Mapped[str] = mapped_column(String, nullable=False, sort_order=4)
    embedding: Mapped[list[float]] = mapped_column(String, nullable=False, sort_order=5)


class ObsidianFields(DocumentFields):
    @classmethod
    @abstractmethod
    def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
        """forward_links に document_id を含む行を絞り込む WHERE 句を返す."""
        raise NotImplementedError


class VaultUtils:
    """ドキュメントを DB に同期するクラス."""

    @staticmethod
    async def sync_docs(
        documents: list[Document],
        sa_engine: AsyncEngine,
        raw_entity: type[DocumentFields],
    ) -> None:
        """テーブルを引数 documents の内容で洗い替えする."""
        rows = [
            {
                "document_id": doc.id,
                "document_metadata": {
                    key: doc.metadata[key]
                    for key in doc.metadata
                    if key not in ("file_path", "document_content_hash")
                },
                "document_content_hash": hashlib.sha256(doc.page_content.encode()).hexdigest(),
                "content": doc.page_content,
                "file_path": doc.metadata["file_path"],
            }
            for doc in documents
        ]
        assert rows is not None
        async with AsyncSession(sa_engine) as session:
            await session.execute(delete(raw_entity))
            await session.execute(insert(raw_entity), rows)
            await session.commit()

    @staticmethod
    async def sync_chunks(
        doc_entity: type[DocumentFields], chk_entity: type, sa_engine: AsyncEngine
    ) -> Sequence[Row]:
        """doc_entity と chunk_entity を document_id で比較し、chunk 側を差分同期する.

        削除対象（hash 変更 / doc から消えた doc_id）のチャンクは実行済み。
        戻り値は再埋め込みが必要な doc_entity の行（新規・変更分）。
        """
        # doc 側: 同じ (document_id, hash) の chunk 行が無い = 新規または変更された doc 行
        matching_chunk = (
            select(chk_entity.document_id)
            .where(chk_entity.document_id == doc_entity.document_id)
            .where(chk_entity.document_content_hash == doc_entity.document_content_hash)
            .correlate(doc_entity)
        )
        diff_stmt = select(doc_entity).where(~matching_chunk.exists())

        # chunk 側: 同じ (document_id, hash) の doc 行が無い = 削除すべき旧チャンク
        matching_doc = (
            select(doc_entity.document_id)
            .where(doc_entity.document_id == chk_entity.document_id)
            .where(doc_entity.document_content_hash == chk_entity.document_content_hash)
            .correlate(chk_entity)
        )
        delete_stmt = delete(chk_entity).where(~matching_doc.exists())

        async with sa_engine.connect() as conn:
            diff_rows = (await conn.execute(diff_stmt)).all()
            await conn.execute(delete_stmt)
            await conn.commit()

        return diff_rows
