from sqlalchemy import MetaData, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class CheckpointBase(DeclarativeBase):
    """app スキーマの既存テーブルへの読み書き用マッピングのみを提供する.

    langgraph-checkpoint-postgres が管理するテーブルのため、SELECT・DELETE で
    実際に使う列のみを定義する（テーブル作成は AsyncPostgresSaver.setup() が担う。
    作成先スキーマは store.py の PostgresStoreConnector.get_psycopg_pool() が
    search_path を app へ固定することで決まる）。
    """

    metadata = MetaData("app")


class CheckpointEntity(CheckpointBase):
    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)


class CheckpointBlobEntity(CheckpointBase):
    __tablename__ = "checkpoint_blobs"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True)
    channel: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[str] = mapped_column(Text, primary_key=True)


class CheckpointWriteEntity(CheckpointBase):
    __tablename__ = "checkpoint_writes"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)
    task_id: Mapped[str] = mapped_column(Text, primary_key=True)
    idx: Mapped[int] = mapped_column(primary_key=True)
