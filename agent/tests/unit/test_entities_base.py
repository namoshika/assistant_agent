from sqlalchemy import JSON, MetaData, insert
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from assistant_agent.entities.base import ChunkFields, DocumentFields, VaultUtils


class _Base(DeclarativeBase):
    metadata = MetaData()


class _DummyDocEntity(_Base, DocumentFields):
    __tablename__ = "doc_raws"

    document_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, sort_order=1)


class _DummyChunkEntity(_Base, ChunkFields):
    __tablename__ = "doc_chunks"


async def test_sync_chunks_01() -> None:
    """Raw と chunk を hash-diff で比較し、差分行の取得と旧チャンクの削除ができること.

    観点1: 変更ドキュメント (doc-changed) が戻り値に含まれ、旧チャンクが削除される
    観点2: 新規ドキュメント (doc-new) が戻り値に含まれる
    観点3: 消えたドキュメント (doc-deleted) の旧チャンクが削除される
    観点4: 変更なしドキュメント (doc-same) は戻り値に含まれず、チャンクも削除されない
    """
    # 試験準備
    raw_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with raw_engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
        await conn.execute(
            insert(_DummyDocEntity),
            [
                {
                    "document_id": "doc-changed",
                    "document_metadata": {},
                    "document_content_hash": "new-hash",
                    "content": "changed content",
                    "file_path": "changed.md",
                },
                {
                    "document_id": "doc-new",
                    "document_metadata": {},
                    "document_content_hash": "hash-new",
                    "content": "new content",
                    "file_path": "new.md",
                },
                {
                    "document_id": "doc-same",
                    "document_metadata": {},
                    "document_content_hash": "hash-same",
                    "content": "same content",
                    "file_path": "same.md",
                },
            ],
        )
        await conn.execute(
            insert(_DummyChunkEntity),
            [
                {
                    "langchain_id": "chunk-changed-old",
                    "content": "old changed content",
                    "document_id": "doc-changed",
                    "document_content_hash": "old-hash",
                },
                {
                    "langchain_id": "chunk-deleted",
                    "content": "deleted content",
                    "document_id": "doc-deleted",
                    "document_content_hash": "hash-deleted",
                },
                {
                    "langchain_id": "chunk-same",
                    "content": "same content chunk",
                    "document_id": "doc-same",
                    "document_content_hash": "hash-same",
                },
            ],
        )

    # 試験実施
    diff_rows = await VaultUtils.sync_chunks(_DummyDocEntity, _DummyChunkEntity, raw_engine)

    # 結果検証
    diff_doc_ids = {row.document_id for row in diff_rows}
    # 観点1
    assert "doc-changed" in diff_doc_ids
    # 観点2
    assert "doc-new" in diff_doc_ids
    # 観点4
    assert "doc-same" not in diff_doc_ids

    async with raw_engine.connect() as conn:
        result = await conn.execute(_DummyChunkEntity.__table__.select())
        remaining_chunk_ids = {row[0] for row in result}
    # 観点1
    assert "chunk-changed-old" not in remaining_chunk_ids
    # 観点3
    assert "chunk-deleted" not in remaining_chunk_ids
    # 観点4
    assert "chunk-same" in remaining_chunk_ids
