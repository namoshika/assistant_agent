from pgvector.sqlalchemy import Vector
from sqlalchemy import MetaData, cast
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement

from assistant_agent.entities import base


class VaultBase(DeclarativeBase):
    metadata = MetaData("assets")


class ChunkBase(DeclarativeBase):
    metadata = MetaData("app")


class DocumentFields(base.DocumentFields):
    """document_metadata=JSONB と backlink_filter を定義."""

    document_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, sort_order=1)


class ChunkFields(base.ChunkFields):
    """PGVectorStore が生成するチャンクテーブルの共通カラム（DDL 上の型に忠実）."""

    langchain_id: Mapped[str] = mapped_column(UUID, primary_key=True, sort_order=0)
    embedding: Mapped[list[float]] = mapped_column(Vector(3072), nullable=False, sort_order=4)
    langchain_metadata: Mapped[dict | None] = mapped_column(JSON, sort_order=5)


class ObsidianFields(DocumentFields, base.ObsidianFields):
    @classmethod
    def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
        """forward_links に document_id を含む行を絞り込む WHERE 句を返す."""
        return cast(cls.document_metadata["forward_links"], JSONB).contains([document_id])


class ObsidianEntity(VaultBase, ObsidianFields):
    """backlink_filter は DocumentFields から継承."""

    __tablename__ = "obsidian_raw"


class SampleEntity(VaultBase, DocumentFields):
    """サンプルデータ用."""

    __tablename__ = "sample_raw"


class ObsidianChunkEntity(ChunkBase, ChunkFields):
    """obsidian_retriever のチャンクテーブル."""

    __tablename__ = "obsidian_vectors"


class SampleChunkEntity(ChunkBase, ChunkFields):
    """sample_retriever のチャンクテーブル."""

    __tablename__ = "sample_vectors"
