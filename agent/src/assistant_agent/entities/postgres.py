from sqlalchemy import MetaData, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement

from assistant_agent.entities import base


class VaultBase(DeclarativeBase):
    metadata = MetaData("assets")


class DocumentFields(base.DocumentFields):
    """document_metadata=JSONB と backlink_filter を定義."""

    document_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, sort_order=1)


class ObsidianFields(DocumentFields, base.ObsidianFields):
    @classmethod
    def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
        """forward_links に document_id を含む行を絞り込む WHERE 句を返す."""
        return cast(cls.document_metadata["forward_links"], JSONB).contains([document_id])


class ObsidianEntity(VaultBase, ObsidianFields):
    """backlink_filter は DocumentFields から継承."""

    __tablename__ = "obsidian_vault_raw"


class WebsiteEntity(VaultBase, DocumentFields):
    """ウェブサイト用."""

    __tablename__ = "website_raw"
