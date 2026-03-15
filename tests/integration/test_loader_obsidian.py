import uuid
import pytest
from collections.abc import Generator
from pathlib import Path
from langchain_core.documents import Document
from sqlalchemy import Engine, MetaData, select, text
from sqlalchemy.orm import DeclarativeBase, Session
from agent_assistant.loader.obsidian import PgVault, VaultLoader
from agent_assistant.model import ObsidianVaultEntity


@pytest.fixture()
def pg_vault_tables(
    sa_engine: Engine,
) -> Generator[tuple[Engine, type[ObsidianVaultEntity]], None, None]:
    """テスト専用の raw テーブルを作成し、テスト後に DROP する。"""
    vault_name = f"test_{uuid.uuid4().hex[:8]}"

    class _TestBase(DeclarativeBase):
        metadata = MetaData("public")

    class _TestRawEntity(_TestBase, ObsidianVaultEntity):
        __tablename__ = f"{vault_name}_raw"

    _TestBase.metadata.create_all(sa_engine)
    yield sa_engine, _TestRawEntity

    with sa_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {vault_name}_raw CASCADE"))
        conn.commit()


class TestVaultLoader:
    @pytest.mark.integration
    def test_load_01(self):
        """load() を呼び出した時、Obsidian Vault ディレクトリ内のドキュメントを Document リストとして正しく返せる。

        観点1: 結果が空でない
        観点2: どの doc も path が絶対パスでなく、空でない
        観点3: どの doc も hash が格納されている
        観点4: forward_links キーが存在しリスト型である
        観点5: forward_links が空でない Document が 1 件以上存在する
        観点6: document_id が doc.id と一致する
        観点7: created / last_modified / last_accessed が存在する
        """
        # 試験準備
        VAULT_PATH = Path("docs/dataset_obsidian/")
        if not VAULT_PATH.exists():
            pytest.fail("Vault が存在しないため失敗")

        # 試験実施
        docs = VaultLoader(VAULT_PATH).load()

        # 結果検証
        # 観点1
        assert len(docs) >= 1
        for doc in docs:
            # 観点2
            path = doc.metadata["path"]
            assert not Path(path).is_absolute(), f"path が絶対パス: {path}"
            assert path != ""
            # 観点3
            assert doc.metadata["hash"] is not None
            # 観点4
            assert (
                "forward_links" in doc.metadata
            ), f"forward_links なし: {doc.metadata.get('path')}"
            assert isinstance(doc.metadata["forward_links"], list)
            # 観点6
            assert doc.id is not None
            assert doc.metadata.get("document_id") == doc.id
            # 観点7
            for dt_key in ("created", "last_modified", "last_accessed"):
                assert dt_key in doc.metadata, f"{dt_key} が存在しない: {path}"
        # 観点5
        has_links = any(len(doc.metadata["forward_links"]) > 0 for doc in docs)
        assert has_links, "forward_links が空でない doc が 1 件もない"


class TestPgVault:
    @pytest.mark.integration
    def test_sync_01(
        self,
        pg_vault_tables: tuple[Engine, type[ObsidianVaultEntity]],
        vault_docs,
    ):
        """sync() を呼び出した時、引数 documents で渡されたドキュメントで raw テーブルを洗い替えできる。

        観点1（1回目 sync）: raw に 3 件が正しく格納される
        観点2（2回目 sync）: A が更新され、C が削除され、B の内容が不変
        """
        # 試験準備
        sa_engine, raw_entity = pg_vault_tables
        note_a, note_b, note_c = vault_docs[:3]
        doc_a_modified = Document(
            id=note_a.id,
            page_content="changed content",
            metadata={**note_a.metadata, "hash": "changed_hash"},
        )

        # 試験実施（1回目）
        PgVault.sync([note_a, note_b, note_c], sa_engine, raw_entity)

        # 結果検証
        # 観点1
        with Session(sa_engine) as session:
            raw_rows = {
                row.document_id: row
                for row in session.scalars(select(raw_entity)).all()
            }
        assert len(raw_rows) == 3
        for note in (note_a, note_b, note_c):
            row = raw_rows[note.id]
            assert row.hash == note.metadata["hash"]
            assert row.content == note.page_content
            assert row.path == note.metadata["path"]

        # 試験実施（2回目: C を削除、A を変更、B はそのまま）
        PgVault.sync([doc_a_modified, note_b], sa_engine, raw_entity)

        # 結果検証
        # 観点2
        with Session(sa_engine) as session:
            raw_rows = {
                row.document_id: row
                for row in session.scalars(select(raw_entity)).all()
            }
        assert raw_rows[note_a.id].hash == "changed_hash"
        assert raw_rows[note_a.id].content == "changed content"
        assert raw_rows[note_b.id].hash == note_b.metadata["hash"]
        assert note_c.id not in raw_rows
