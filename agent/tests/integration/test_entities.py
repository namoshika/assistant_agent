import pytest
from llama_index.core import Document
from sqlalchemy import Engine, MetaData, insert, select
from sqlalchemy.orm import DeclarativeBase, Session

from assistant_agent.entities import duckdb, postgres
from assistant_agent.entities.base import VaultUtils
from assistant_agent.utils.store_context import DuckDBStoreContext, PostgresStoreContext


@pytest.fixture()
def pg_backlink_entity(pg_cxt: PostgresStoreContext):
    """PostgreSQL 用バックリンクフィルタ検証テーブル."""
    engine = pg_cxt.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestEntity(_TestBase, postgres.ObsidianFields):
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

    class _TestEntity(_TestBase, duckdb.ObsidianFields):
        __tablename__ = "test_backlink_filter_dk"

    _TestBase.metadata.create_all(engine)
    yield _TestEntity, engine
    _TestBase.metadata.drop_all(engine)


class TestObsidianFields:
    @pytest.mark.integration
    def test_backlink_filter_01(
        self, pg_backlink_entity: tuple[type[postgres.ObsidianFields], Engine]
    ):
        """postgres.DocumentFields.backlink_filter が forward_links でフィルタできること.

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
                        "file_path": "A.md",
                    },
                    {
                        "document_id": "id-B",
                        "document_metadata": {"forward_links": []},
                        "content": "B",
                        "file_path": "B.md",
                    },
                ],
            )
            session.commit()

        # 試験実施
        with Session(engine) as session:
            rows = session.scalars(
                select(entity_cls).where(entity_cls.backlink_filter("id-B"))
            ).all()

        # 結果検証
        # 観点1
        assert len(rows) == 1
        assert rows[0].document_id == "id-A"

    @pytest.mark.integration
    def test_backlink_filter_02(
        self, dk_backlink_entity: tuple[type[duckdb.ObsidianFields], Engine]
    ):
        """duckdb.DocumentFields.backlink_filter が forward_links でフィルタできること.

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
                        "file_path": "A.md",
                    },
                    {
                        "document_id": "id-B",
                        "document_metadata": {"forward_links": []},
                        "content": "B",
                        "file_path": "B.md",
                    },
                ],
            )
            session.commit()

        # 試験実施
        with Session(engine) as session:
            rows = session.scalars(
                select(entity_cls).where(entity_cls.backlink_filter("id-B"))
            ).all()

        # 結果検証
        # 観点1
        assert len(rows) == 1
        assert rows[0].document_id == "id-A"


class TestVaultUtils:
    @pytest.mark.integration
    def test_sync_01(
        self,
        pg_cxt: PostgresStoreContext,
        pg_entity_obs: type[postgres.DocumentFields],
        docs_obs: list[Document],
    ):
        """sync() を呼び出した時、引数として渡されたドキュメントで raw テーブルを洗い替えできる.

        観点1（1回目 sync）: raw に 3 件が正しく格納される
        観点2（2回目 sync）: A が更新され、C が削除され、B の内容が不変
        """
        # 試験準備
        raw_entity = pg_entity_obs
        sa_engine = pg_cxt.get_engine()
        note_a, note_b, note_c = docs_obs[:3]
        doc_a_modified = Document(
            id_=note_a.id_,
            text="changed content",
            metadata=note_a.metadata,
        )

        # 試験実施（1回目）
        VaultUtils.sync([note_a, note_b, note_c], sa_engine, raw_entity)

        # 結果検証
        # 観点1
        with Session(sa_engine) as session:
            raw_rows = {row.document_id: row for row in session.scalars(select(raw_entity)).all()}
        assert len(raw_rows) == 3
        for note in (note_a, note_b, note_c):
            row = raw_rows[note.id_]
            assert row.content == note.text
            assert row.file_path == note.metadata["file_path"]

        # 試験実施（2回目: C を削除、A を変更、B はそのまま）
        VaultUtils.sync([doc_a_modified, note_b], sa_engine, raw_entity)

        # 結果検証
        # 観点2
        with Session(sa_engine) as session:
            raw_rows = {row.document_id: row for row in session.scalars(select(raw_entity)).all()}
        assert raw_rows[note_a.id_].content == "changed content"
        assert note_c.id_ not in raw_rows
