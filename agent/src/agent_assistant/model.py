from sqlalchemy import MetaData, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ObsidianVaultEntity:
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False, unique=True)


class ObsidianVaultBase(DeclarativeBase):
    metadata = MetaData("public")
    pass


class ObsidianVaultRawEntity(ObsidianVaultBase, ObsidianVaultEntity):
    __tablename__ = "obsidian_vault_raw"
    pass
