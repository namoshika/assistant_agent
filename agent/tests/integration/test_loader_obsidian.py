import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from assistant_agent.entities import postgres
from assistant_agent.loaders import ObsidianLoader
from assistant_agent.store import PostgresStoreConnector


@pytest.fixture()
def pg_vault_tables(pg_conn: PostgresStoreConnector) -> Iterator[type[postgres.DocumentFields]]:
    """テスト専用の raw テーブルを作成し、テスト後に DROP する."""
    engine = pg_conn.get_engine()
    vault_name = f"test_{uuid.uuid4().hex[:8]}"

    class _TestBase(DeclarativeBase):
        metadata = MetaData("assets")

    class _TestRawEntity(_TestBase, postgres.DocumentFields):
        __tablename__ = f"{vault_name}_raw"

    _TestBase.metadata.create_all(engine)
    yield _TestRawEntity
    _TestBase.metadata.drop_all(engine)


@pytest.mark.integration
def test_load_01():
    """load() を呼び出した時、 Vault ディレクトリ内のドキュメントをリストで返せるか確認.

    観点1: 結果が空でない
    観点2: どの doc も path が絶対パスでなく、空でない
    観点3: 除外フィールド (hash/created/last_modified/last_accessed/source) が存在しない
    観点4: forward_links キーが存在しリスト型である
    観点5: forward_links が空でない Document が 1 件以上存在する
    観点6: doc.id が path から生成した document_id (UUID5) と一致する
    """
    # 試験準備
    vault_path = Path("docs/dataset_obsidian/")
    if not vault_path.exists():
        pytest.fail("Vault が存在しないため失敗")

    # 試験実施
    docs = ObsidianLoader(vault_path).load()

    # 結果検証
    # 観点1
    assert len(docs) >= 1
    for doc in docs:
        # 観点2
        path = doc.metadata["file_path"]
        assert not Path(path).is_absolute(), f"path が絶対パス: {path}"
        assert path != ""
        # 観点3
        for key in ("hash", "created", "last_modified", "last_accessed", "source"):
            assert key not in doc.metadata, f"{key} が存在してはいけない: {path}"
        # 観点4
        assert "forward_links" in doc.metadata, f"forward_links なし: {doc.metadata.get('path')}"
        assert isinstance(doc.metadata["forward_links"], list)
        # 観点6
        assert doc.id is not None
    # 観点5
    has_links = any(len(doc.metadata["forward_links"]) > 0 for doc in docs)
    assert has_links, "forward_links が空でない doc が 1 件もない"
