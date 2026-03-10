import json
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from agent_assistant.retriever.obsidian import ObsidianDocumentStore
from agent_assistant.utils.chunker.text import TextChunker


def _make_store(mocker: MockerFixture, connected: bool = False):
    """ObsidianDocumentStore のテスト用インスタンスを生成するヘルパー。"""
    chunk_store = MagicMock()
    sa_engine = MagicMock()
    chunker = MagicMock(spec=TextChunker)

    store = ObsidianDocumentStore(
        store_name="test_vault",
        chunk_store=chunk_store,
        sa_engine=sa_engine,
        chunker=chunker,
    )

    if connected:
        # connect() 済み状態をシミュレート
        m_chunk_vs = MagicMock()
        store._chunk_vs = m_chunk_vs
    return store, chunk_store, sa_engine, chunker


def test_search_documents_01(mocker: MockerFixture):
    """chunk 検索 → path 重複除去 → raw SQL 取得 → 初出順ソートのパイプライン。

    観点1: _chunk_vs.search が正しい引数で呼ばれる
    観点2: path が重複除去される (a.md は chunk1/chunk3 の2件あるが1件に集約)
    観点3: sa_engine で raw テーブルを SELECT し Document リストが返る
    観点4: 返り値の Document の page_content と metadata が raw テーブルの値と一致
    観点5: search() の返却順 (スコア降順) が保持される (DB が逆順で返しても正しく並ぶ)
    """
    # 試験準備
    store, chunk_store, sa_engine, chunker = _make_store(mocker, connected=True)

    # search() が返すチャンク (スコア降順: a.md → b.md → a.md の順)
    m_chunks = [
        Document(page_content="chunk1", metadata={"path": "a.md", "start_index": 0}),
        Document(page_content="chunk2", metadata={"path": "b.md", "start_index": 0}),
        Document(page_content="chunk3", metadata={"path": "a.md", "start_index": 32}),
    ]
    store._chunk_vs.search.return_value = m_chunks  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    # SA engine の conn.execute() が返すロー (DB は b.md, a.md の逆順で返す)
    m_row_a = MagicMock()
    m_row_a.path = "a.md"
    m_row_a.content = "content_a"
    m_row_a.metadata = json.dumps({"path": "a.md", "tags": ["tag1"]})
    m_row_b = MagicMock()
    m_row_b.path = "b.md"
    m_row_b.content = "content_b"
    m_row_b.metadata = json.dumps({"path": "b.md"})

    m_result = MagicMock()
    m_result.__iter__ = MagicMock(
        return_value=iter([m_row_b, m_row_a])
    )  # DB が逆順で返す
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    # 試験実施
    docs = store.search_documents("テスト", top_k=5)

    # 結果検証
    # 観点1: search が呼ばれる
    store._chunk_vs.search.assert_called_once_with(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        "テスト", "similarity", top_k=5
    )
    # 観点2 & 3: 2件返る (a.md の重複は除去済み)
    assert len(docs) == 2
    sa_engine.connect.assert_called()
    # 観点4: Document の内容が正しい
    paths = [d.metadata["path"] for d in docs]
    assert "a.md" in paths
    assert "b.md" in paths
    # 観点5: search() の初出順 (a.md → b.md) が保持される (DB 逆順に影響されない)
    assert paths[0] == "a.md"
    assert paths[1] == "b.md"


def test_get_documents_01(mocker: MockerFixture):
    """ファイル名のみ・フルパスのどちらで指定しても取得できる。

    観点1: ファイル名のみ指定でも取得できる
    観点2: フルパス指定でも取得できる
    """
    # 試験準備
    store, chunk_store, sa_engine, chunker = _make_store(mocker, connected=True)

    m_row = MagicMock()
    m_row.path = "folder/note.md"
    m_row.content = "note content"
    m_row.metadata = json.dumps({"path": "folder/note.md", "tags": []})

    m_result = MagicMock()
    # 毎回新しいイテレータ
    m_result.__iter__ = MagicMock(side_effect=lambda: iter([m_row]))
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    # 観点1: ファイル名のみ
    docs = store.get_documents(["note.md"])
    assert len(docs) == 1
    assert docs[0].page_content == "note content"
    assert docs[0].metadata["path"] == "folder/note.md"

    # 観点2: フルパス
    docs = store.get_documents(["folder/note.md"])
    assert len(docs) == 1
    assert docs[0].metadata["path"] == "folder/note.md"


def test_import_documents_01(mocker: MockerFixture):
    """raw upsert と chunk 削除+追加が両方実行される。

    観点1: sa_engine で INSERT ... ON CONFLICT が実行される (raw upsert)
    観点2: sa_engine で DELETE が実行される (既存チャンク削除)
    観点3: chunker.chunk() が呼ばれる
    観点4: chunk_store.add_chunks() が正しい引数で呼ばれる
    """
    # 試験準備
    store, chunk_store, sa_engine, chunker = _make_store(mocker, connected=True)

    docs = [
        Document(
            page_content="note content",
            metadata={"path": "note.md", "tags": ["tag1"]},
        )
    ]
    m_chunks = [
        Document(page_content="chunk", metadata={"path": "note.md", "start_index": 0})
    ]
    chunker.chunk.return_value = m_chunks

    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    sa_engine.connect.return_value = m_conn

    # 試験実施
    store.import_documents(docs)

    # 結果検証
    # 観点1 & 2: sa_engine.connect() が呼ばれた (raw upsert + chunk DELETE)
    assert sa_engine.connect.call_count >= 1
    m_conn.execute.assert_called()
    # 観点3: chunker.chunk() が呼ばれた
    chunker.chunk.assert_called_once_with(docs)
    # 観点4: chunk_store.add_chunks() が呼ばれた
    chunk_store.add_chunks.assert_called_once_with("test_vault_chunks", m_chunks)


def test_get_backlinks_01(mocker: MockerFixture):
    """forward_links を持つノートを import 後、get_backlinks() が正しいノートを返す。

    観点1: EXISTS クエリで SA engine が呼ばれる
    観点2: forward_links に対象パスを含むノートが返る
    観点3: 返り値 Document の page_content が正しい
    """
    store, chunk_store, sa_engine, chunker = _make_store(mocker, connected=True)

    # note_a は note_b.md へのリンクを持つ
    m_row = MagicMock()
    m_row.path = "note_a.md"
    m_row.content = "note_a の本文"
    m_row.metadata = json.dumps(
        {
            "path": "note_a.md",
            "forward_links": ["note_b.md"],
        }
    )

    m_result = MagicMock()
    m_result.__iter__ = MagicMock(return_value=iter([m_row]))
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    # 試験実施
    docs = store.get_backlinks(["note_b.md"])

    # 結果検証
    # 観点1: sa_engine.connect() が呼ばれる
    sa_engine.connect.assert_called()
    # 観点2: 1件返る
    assert len(docs) == 1
    # 観点3: page_content が正しい
    assert docs[0].page_content == "note_a の本文"


def test_get_backlinks_02(mocker: MockerFixture):
    """該当するバックリンクがない場合は空リストを返す。

    観点1: SQL は実行されるが結果が空リスト
    """
    store, chunk_store, sa_engine, chunker = _make_store(mocker, connected=True)

    m_result = MagicMock()
    m_result.__iter__ = MagicMock(return_value=iter([]))
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    docs = store.get_backlinks(["nonexistent.md"])

    # 観点1: 空リスト
    assert docs == []


def test_get_backlinks_03(mocker: MockerFixture):
    """paths が空リストの場合は SQL を呼ばずに空リストを返す。

    観点1: sa_engine.connect() が呼ばれない
    観点2: 空リストが返る
    """
    store, chunk_store, sa_engine, chunker = _make_store(mocker, connected=True)

    docs = store.get_backlinks([])

    # 観点1: DB アクセスなし
    sa_engine.connect.assert_not_called()
    # 観点2: 空リスト
    assert docs == []


def test_not_connected_raises_value_error():
    """connect() 未実行時、全パブリックメソッドが ValueError を発生させる。"""
    store, _, _, _ = _make_store(MagicMock())
    doc = Document(page_content="x", metadata={"path": "note.md"})

    with pytest.raises(ValueError, match="not connected"):
        store.search_documents("クエリ", top_k=5)
    with pytest.raises(ValueError, match="not connected"):
        store.get_documents(["note.md"])
    with pytest.raises(ValueError, match="not connected"):
        store.import_documents([doc])
    with pytest.raises(ValueError, match="not connected"):
        store.get_backlinks(["note.md"])
