from sqlalchemy import MetaData, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.types import JSON

from assistant_agent.entities import base


class ObsidianVaultBase(DeclarativeBase):
    metadata = MetaData("assets")


class ObsidianVaultEntity(base.ObsidianVaultEntity):
    """DuckDB 用ミックスイン. document_metadata=JSON と backlink_filter を定義."""

    document_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, sort_order=1)

    @classmethod
    def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
        """forward_links に document_id を含む行を絞り込む WHERE 句を返す."""
        return text(
            "list_contains(from_json(document_metadata->'forward_links', '\"VARCHAR[]\"'), :target)"
        ).bindparams(target=document_id)  # pyright: ignore[reportReturnType]


class ObsidianVaultRawEntity(ObsidianVaultBase, ObsidianVaultEntity):
    """DuckDB 用テーブルクラス. backlink_filter は ObsidianVaultEntity から継承."""

    __tablename__ = "obsidian_vault_raw"
