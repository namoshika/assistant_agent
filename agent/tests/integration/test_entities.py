import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import MetaData, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from assistant_agent.entities import postgres
from assistant_agent.entities.base import VaultUtils
from assistant_agent.store import PostgresStoreConnector


@pytest.fixture()
async def pg_backlink_entity(pg_conn: PostgresStoreConnector):
    """PostgreSQL 用バックリンクフィルタ検証テーブル."""
    engine = pg_conn.get_engine()

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestEntity(_TestBase, postgres.ObsidianFields):
        __tablename__ = "test_backlink_filter_pg"

    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.create_all)
    yield _TestEntity, engine
    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.drop_all)


class TestObsidianFields:
    @pytest.mark.integration
    async def test_backlink_filter_01(
        self, pg_backlink_entity: tuple[type[postgres.ObsidianFields], AsyncEngine]
    ):
        """postgres.DocumentFields.backlink_filter が forward_links でフィルタできること.

        観点1: forward_links に document_id を含む行のみ返ること
        """
        entity_cls, engine = pg_backlink_entity

        # 試験準備
        async with AsyncSession(engine) as session:
            await session.execute(
                insert(entity_cls),
                [
                    {
                        "document_id": "id-A",
                        "document_metadata": {"forward_links": ["id-B"]},
                        "document_content_hash": "hash-a",
                        "content": "A",
                        "file_path": "A.md",
                    },
                    {
                        "document_id": "id-B",
                        "document_metadata": {"forward_links": []},
                        "document_content_hash": "hash-b",
                        "content": "B",
                        "file_path": "B.md",
                    },
                ],
            )
            await session.commit()

        # 試験実施
        async with AsyncSession(engine) as session:
            rows = (
                await session.scalars(select(entity_cls).where(entity_cls.backlink_filter("id-B")))
            ).all()

        # 結果検証
        # 観点1
        assert len(rows) == 1
        assert rows[0].document_id == "id-A"


class TestVaultUtils:
    @pytest.mark.integration
    async def test_sync_docs_01(
        self,
        pg_conn: PostgresStoreConnector,
        pg_entity_obs: type[postgres.DocumentFields],
        docs_obs: list[Document],
    ):
        """sync() を呼び出した時、引数として渡されたドキュメントで raw テーブルを洗い替えできる.

        観点1（1回目 sync）: raw に 3 件が正しく格納される
        観点2（2回目 sync）: A が更新され、C が削除され、B の内容が不変
        """
        # 試験準備
        raw_entity = pg_entity_obs
        sa_engine = pg_conn.get_engine()
        note_a, note_b, note_c = docs_obs[:3]
        doc_a_modified = Document(
            id=note_a.id,
            page_content="changed content",
            metadata=note_a.metadata,
        )

        # 試験実施（1回目）
        await VaultUtils.sync_docs([note_a, note_b, note_c], sa_engine, raw_entity)

        # 結果検証
        # 観点1
        async with AsyncSession(sa_engine) as session:
            raw_rows = {
                row.document_id: row for row in (await session.scalars(select(raw_entity))).all()
            }
        assert len(raw_rows) == 3
        for note in (note_a, note_b, note_c):
            assert note.id is not None
            row = raw_rows[note.id]
            assert row.content == note.page_content
            assert row.file_path == note.metadata["file_path"]
            expected_hash = hashlib.sha256(note.page_content.encode()).hexdigest()
            assert row.document_content_hash == expected_hash

        # 試験実施（2回目: C を削除、A を変更、B はそのまま）
        await VaultUtils.sync_docs([doc_a_modified, note_b], sa_engine, raw_entity)

        # 結果検証
        # 観点2
        async with AsyncSession(sa_engine) as session:
            raw_rows = {
                row.document_id: row for row in (await session.scalars(select(raw_entity))).all()
            }
        assert note_a.id is not None
        assert raw_rows[note_a.id].content == "changed content"
        assert note_c.id not in raw_rows
        expected_hash = hashlib.sha256(b"changed content").hexdigest()
        assert raw_rows[note_a.id].document_content_hash == expected_hash

    @pytest.mark.integration
    async def test_sync_chunks_01(
        self,
        pg_conn: PostgresStoreConnector,
        pg_entity_obs: type[postgres.DocumentFields],
        pg_entity_chk: type,
    ):
        """hash-diff により変更・新規・削除を検知し、実際に PostgreSQL へ反映されること.

        docs/specs/20260314_sql_orm/07_chunk_content_hash.py Part 2 の検証シナリオを踏襲。

        観点1: 初回同期で doc-A・doc-B が差分として返り、チャンクが登録される
        観点2: doc-A の内容変更 + doc-C 新規追加後、doc-A・doc-C のみ差分として返り、
            doc-A は新内容のみ・doc-C も登録される（doc-B は差分に含まれない）
        観点3: doc-B を渡さなくなった後、差分は空になり、doc-B のチャンクが削除される
        """
        # 試験準備
        obs_entity = pg_entity_obs
        chk_entity = pg_entity_chk
        sa_engine = pg_conn.get_engine()
        embed_dim = chk_entity.__table__.c.embedding.type.dim
        emb = DeterministicFakeEmbedding(size=embed_dim)
        store = await PGVectorStore.create(
            engine=PGEngine.from_engine(sa_engine),
            embedding_service=emb,
            table_name=chk_entity.__tablename__,
            schema_name=chk_entity.metadata.schema,
            ignore_metadata_columns=["langchain_metadata"],
        )

        doc_a = Document(id="docA", page_content="A content v1", metadata={"file_path": "a.md"})
        doc_b = Document(id="docB", page_content="B content v1", metadata={"file_path": "b.md"})

        # 試験実施（1回目: 新規2件）
        await VaultUtils.sync_docs([doc_a, doc_b], sa_engine, obs_entity)
        diff_rows = await VaultUtils.sync_chunks(obs_entity, chk_entity, sa_engine)

        # 結果検証
        # 観点1
        assert {row.document_id for row in diff_rows} == {"docA", "docB"}
        await store.aadd_documents(
            [
                Document(
                    page_content=row.content,
                    metadata={
                        "document_id": row.document_id,
                        "document_content_hash": row.document_content_hash,
                    },
                )
                for row in diff_rows
            ]
        )
        results = await store.asimilarity_search("content", k=10)
        assert {d.metadata["document_id"] for d in results} == {"docA", "docB"}

        # 試験実施（2回目: docA 変更、docB 不変、docC 新規）
        doc_a_changed = Document(
            id="docA", page_content="A content v2 CHANGED", metadata={"file_path": "a.md"}
        )
        doc_c = Document(id="docC", page_content="C content v1", metadata={"file_path": "c.md"})
        await VaultUtils.sync_docs([doc_a_changed, doc_b, doc_c], sa_engine, obs_entity)
        diff_rows = await VaultUtils.sync_chunks(obs_entity, chk_entity, sa_engine)

        # 観点2
        assert {row.document_id for row in diff_rows} == {"docA", "docC"}
        await store.aadd_documents(
            [
                Document(
                    page_content=row.content,
                    metadata={
                        "document_id": row.document_id,
                        "document_content_hash": row.document_content_hash,
                    },
                )
                for row in diff_rows
            ]
        )
        results = await store.asimilarity_search("content", k=10)
        by_doc = {d.metadata["document_id"]: d.page_content for d in results}
        assert by_doc == {
            "docA": "A content v2 CHANGED",
            "docB": "B content v1",
            "docC": "C content v1",
        }

        # 試験実施（3回目: docB を渡さない = vault から削除）
        await VaultUtils.sync_docs([doc_a_changed, doc_c], sa_engine, obs_entity)
        diff_rows = await VaultUtils.sync_chunks(obs_entity, chk_entity, sa_engine)

        # 観点3
        assert diff_rows == []
        results = await store.asimilarity_search("content", k=10)
        assert {d.metadata["document_id"] for d in results} == {"docA", "docC"}
