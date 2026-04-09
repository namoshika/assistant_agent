import pytest
from sqlalchemy import MetaData, insert, select
from sqlalchemy.orm import DeclarativeBase, Session

from agent_assistant.entities import duckdb, postgres
from agent_assistant.utils.store_factory import DuckDBStoreContext, PostgresStoreContext


@pytest.fixture()
def pg_backlink_entity(pg_cxt: PostgresStoreContext):
    """PostgreSQL 用バックリンクフィルタ検証テーブル."""
    engine = pg_cxt.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestEntity(_TestBase, postgres.ObsidianVaultEntity):
        __tablename__ = "test_backlink_filter_pg"

    _TestBase.metadata.create_all(engine)
    yield _TestEntity, engine
    _TestBase.metadata.drop_all(engine)


@pytest.fixture()
def dk_backlink_entity(dk_cxt: DuckDBStoreContext):
    """DuckDB 用バックリンクフィルタ検証テーブル."""
    engine = dk_cxt.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData()

    class _TestEntity(_TestBase, duckdb.ObsidianVaultEntity):
        __tablename__ = "test_backlink_filter_dk"

    _TestBase.metadata.create_all(engine)
    yield _TestEntity, engine
    _TestBase.metadata.drop_all(engine)


@pytest.mark.integration
def test_backlink_filter_01(pg_backlink_entity):
    """postgres.ObsidianVaultEntity.backlink_filter が forward_links でフィルタできること.

    観点1: forward_links に document_id を含む行のみ返ること
    """
    entity_cls, engine = pg_backlink_entity

    # 試験準備
    with Session(engine) as session:
        session.execute(
            insert(entity_cls),
            [
                {
                    "document_id": "id-A",
                    "document_metadata": {"forward_links": ["id-B"]},
                    "content": "A",
                    "path": "A.md",
                },
                {
                    "document_id": "id-B",
                    "document_metadata": {"forward_links": []},
                    "content": "B",
                    "path": "B.md",
                },
            ],
        )
        session.commit()

    # 試験実施
    with Session(engine) as session:
        rows = session.scalars(select(entity_cls).where(entity_cls.backlink_filter("id-B"))).all()

    # 結果検証
    # 観点1
    assert len(rows) == 1
    assert rows[0].document_id == "id-A"


@pytest.mark.integration
def test_backlink_filter_02(dk_backlink_entity):
    """duckdb.ObsidianVaultEntity.backlink_filter が forward_links でフィルタできること.

    観点1: forward_links に document_id を含む行のみ返ること
    """
    entity_cls, engine = dk_backlink_entity

    # 試験準備
    with Session(engine) as session:
        session.execute(
            insert(entity_cls),
            [
                {
                    "document_id": "id-A",
                    "document_metadata": {"forward_links": ["id-B"]},
                    "content": "A",
                    "path": "A.md",
                },
                {
                    "document_id": "id-B",
                    "document_metadata": {"forward_links": []},
                    "content": "B",
                    "path": "B.md",
                },
            ],
        )
        session.commit()

    # 試験実施
    with Session(engine) as session:
        rows = session.scalars(select(entity_cls).where(entity_cls.backlink_filter("id-B"))).all()

    # 結果検証
    # 観点1
    assert len(rows) == 1
    assert rows[0].document_id == "id-A"
