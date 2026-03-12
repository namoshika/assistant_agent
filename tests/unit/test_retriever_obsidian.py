import json
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from agent_assistant.retriever.obsidian import (
    ObsidianDocumentStore,
    path_to_document_id,
)
from agent_assistant.utils.chunker.text import TextChunker

_DOC_ID_A = path_to_document_id("a.md")
_DOC_ID_B = path_to_document_id("b.md")
_DOC_ID_NOTE = path_to_document_id("folder/note.md")
_DOC_ID_NOTE_A = path_to_document_id("note_a.md")
_DOC_ID_NOTE_B = path_to_document_id("note_b.md")


def _make_store(connected: bool = False):
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


def test_connect_01():
    """connect() が DB に SQL を実行し、2回目以降は何もしない。

    観点1: conn.execute() が1回呼ばれる (CREATE TABLE 文)
    観点2: conn.commit() が1回呼ばれる
    観点3: chunk_store.get_vectorstore() が1回呼ばれ _chunk_vs に設定される
    観点4: 2回目の connect() 後も execute / get_vectorstore は1回のまま
    """
    # 試験準備
    store, chunk_store, sa_engine, _ = _make_store()
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    sa_engine.connect.return_value = m_conn

    # 試験実施 (1回目)
    store.connect()

    # 結果検証 (1回目)
    # 観点1
    m_conn.execute.assert_called_once()
    # 観点2
    m_conn.commit.assert_called_once()
    # 観点3
    chunk_store.get_vectorstore.assert_called_once_with(store._chunk_table)
    assert store._chunk_vs is chunk_store.get_vectorstore.return_value

    # 試験実施 (2回目)
    store.connect()

    # 結果検証 (2回目)
    # 観点4: 呼び出し回数が増えない
    m_conn.execute.assert_called_once()
    chunk_store.get_vectorstore.assert_called_once()


def test_search_documents_01():
    """chunk 検索 → document_id 重複除去 → raw SQL 取得 → 初出順ソートのパイプライン。

    観点1: _chunk_vs.search が正しい引数で呼ばれる
    観点2: document_id が重複除去される (a.md は chunk1/chunk3 の2件あるが1件に集約)
    観点3: sa_engine で raw テーブルを SELECT し Document リストが返る
    観点4: 返り値の Document の page_content と metadata が raw テーブルの値と一致
    観点5: search() の返却順 (スコア降順) が保持される (DB が逆順で返しても正しく並ぶ)
    """
    # 試験準備
    store, _, sa_engine, _ = _make_store(connected=True)

    # search() が返すチャンク (スコア降順: a.md → b.md → a.md の順)
    m_chunks = [
        Document(
            page_content="chunk1",
            metadata={"document_id": _DOC_ID_A, "path": "a.md", "start_index": 0},
        ),
        Document(
            page_content="chunk2",
            metadata={"document_id": _DOC_ID_B, "path": "b.md", "start_index": 0},
        ),
        Document(
            page_content="chunk3",
            metadata={"document_id": _DOC_ID_A, "path": "a.md", "start_index": 32},
        ),
    ]
    store._chunk_vs.search.return_value = m_chunks  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    # SA engine の conn.execute() が返すロー (DB は b.md, a.md の逆順で返す)
    m_row_a = MagicMock()
    m_row_a.document_id = _DOC_ID_A
    m_row_a.path = "a.md"
    m_row_a.content = "content_a"
    m_row_a.metadata = json.dumps(
        {"path": "a.md", "document_id": _DOC_ID_A, "tags": ["tag1"]}
    )
    m_row_b = MagicMock()
    m_row_b.document_id = _DOC_ID_B
    m_row_b.path = "b.md"
    m_row_b.content = "content_b"
    m_row_b.metadata = json.dumps({"path": "b.md", "document_id": _DOC_ID_B})

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
    # 観点1
    store._chunk_vs.search.assert_called_once_with(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        "テスト", "similarity", top_k=5
    )
    # 観点2 & 3
    assert len(docs) == 2
    sa_engine.connect.assert_called()
    # 観点4
    assert docs[0].page_content == "content_a"
    assert docs[0].metadata["path"] == "a.md"
    assert docs[1].page_content == "content_b"
    # 観点5: DB 逆順 (b→a) に関わらず search 初出順 (a→b) で返る
    doc_ids = [d.metadata["document_id"] for d in docs]
    assert doc_ids[0] == _DOC_ID_A
    assert doc_ids[1] == _DOC_ID_B


def test_get_documents_by_ids_01():
    """document_id の完全一致で Document が返る。

    観点1: sa_engine で DB アクセスが行われる
    観点2: 返り値の Document に document_id メタデータが含まれる
    """
    # 試験準備
    store, _, sa_engine, _ = _make_store(connected=True)
    m_row = MagicMock()
    m_row.document_id = _DOC_ID_NOTE
    m_row.path = "folder/note.md"
    m_row.content = "note content"
    m_row.metadata = json.dumps({"path": "folder/note.md"})

    m_result = MagicMock()
    m_result.__iter__ = MagicMock(return_value=iter([m_row]))
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    # 試験実施
    docs = store.get_documents_by_ids([_DOC_ID_NOTE])

    # 結果検証
    # 観点1
    sa_engine.connect.assert_called()
    # 観点2
    assert len(docs) == 1
    assert docs[0].metadata["document_id"] == _DOC_ID_NOTE


def test_get_document_by_path_01():
    """ファイル名のみ・フルパスのどちらで指定しても取得できる。

    観点1: ファイル名のみ指定でも取得できる
    観点2: フルパス指定でも取得できる
    観点3: 返り値の Document に document_id が含まれる
    """
    # 試験準備
    store, _, sa_engine, _ = _make_store(connected=True)

    m_row = MagicMock()
    m_row.document_id = _DOC_ID_NOTE
    m_row.path = "folder/note.md"
    m_row.content = "note content"
    m_row.metadata = json.dumps(
        {"path": "folder/note.md", "document_id": _DOC_ID_NOTE, "tags": []}
    )

    m_result = MagicMock()
    # 毎回新しいイテレータ
    m_result.__iter__ = MagicMock(side_effect=lambda: iter([m_row]))
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    # 観点1: ファイル名のみ
    docs = store.get_document_by_path("note.md")
    assert len(docs) == 1
    assert docs[0].page_content == "note content"
    assert docs[0].metadata["path"] == "folder/note.md"
    assert docs[0].metadata["document_id"] == _DOC_ID_NOTE

    # 観点2: フルパス
    docs = store.get_document_by_path("folder/note.md")
    assert len(docs) == 1
    assert docs[0].metadata["path"] == "folder/note.md"


def test_import_documents_01():
    """raw upsert と chunk 削除+追加が両方実行される。

    観点1: sa_engine で INSERT ... ON CONFLICT が実行される (raw upsert)
    観点2: sa_engine で DELETE が実行される (既存チャンク削除)
    観点3: chunker.chunk() が呼ばれる
    観点4: chunk_store.add_chunks() が正しい引数で呼ばれる
    観点5: import 後のドキュメントに document_id が付与されている
    """
    # 試験準備
    store, chunk_store, sa_engine, chunker = _make_store(connected=True)
    docs = [
        Document(
            page_content="note content",
            metadata={"path": "note.md", "tags": ["tag1"]},
        )
    ]
    expected_doc_id = path_to_document_id("note.md")
    m_chunks = [
        Document(
            page_content="chunk",
            metadata={
                "document_id": expected_doc_id,
                "path": "note.md",
                "start_index": 0,
            },
        )
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
    # 観点5: document_id が付与されている
    assert docs[0].metadata["document_id"] == expected_doc_id


def test_get_backlinks_01():
    """forward_links を持つノートを import 後、get_backlinks() が正しいノートを返す。

    観点1: EXISTS クエリで SA engine が呼ばれる
    観点2: forward_links に対象 document_id を含むノートが返る
    観点3: 返り値 Document の page_content が正しい
    """
    # 試験準備
    store, _, sa_engine, _ = _make_store(connected=True)

    # note_a は note_b.md へのリンクを持つ
    m_row = MagicMock()
    m_row.document_id = _DOC_ID_NOTE_A
    m_row.path = "note_a.md"
    m_row.content = "note_a の本文"
    m_row.metadata = json.dumps(
        {
            "path": "note_a.md",
            "document_id": _DOC_ID_NOTE_A,
            "forward_links": [_DOC_ID_NOTE_B],
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
    docs = store.get_backlinks(_DOC_ID_NOTE_B)

    # 結果検証
    # 観点1: sa_engine.connect() が呼ばれる
    sa_engine.connect.assert_called()
    # 観点2: 1件返る
    assert len(docs) == 1
    # 観点3: page_content が正しい
    assert docs[0].page_content == "note_a の本文"


def test_get_backlinks_02():
    """該当するバックリンクがない場合は空リストを返す。

    観点1: SQL は実行されるが結果が空リスト
    """
    # 試験準備
    store, _, sa_engine, _ = _make_store(connected=True)

    m_result = MagicMock()
    m_result.__iter__ = MagicMock(return_value=iter([]))
    m_conn = MagicMock()
    m_conn.__enter__ = MagicMock(return_value=m_conn)
    m_conn.__exit__ = MagicMock(return_value=False)
    m_conn.execute.return_value = m_result
    sa_engine.connect.return_value = m_conn

    # 試験実施
    docs = store.get_backlinks(path_to_document_id("nonexistent.md"))

    # 結果検証
    # 観点1
    assert docs == []


def test_get_backlinks_03():
    """document_id が空文字列の場合は ValueError を raise する。

    観点1: sa_engine.connect() が呼ばれない
    観点2: ValueError が raise される
    """
    # 試験準備
    store, _, sa_engine, _ = _make_store(connected=True)

    # 試験実施, 結果検証
    with pytest.raises(ValueError):
        store.get_backlinks("")
    sa_engine.connect.assert_not_called()


def test_not_connected_raises_value_error_01():
    """connect() 未実行時、全パブリックメソッドが ValueError を発生させる。"""
    # 試験準備
    store, _, _, _ = _make_store()
    doc = Document(page_content="x", metadata={"path": "note.md"})

    # 試験実施, 結果検証
    with pytest.raises(ValueError, match="not connected"):
        store.search_documents("クエリ", top_k=5)
    with pytest.raises(ValueError, match="not connected"):
        store.get_documents_by_ids([path_to_document_id("note.md")])
    with pytest.raises(ValueError, match="not connected"):
        store.get_document_by_path("note.md")
    with pytest.raises(ValueError, match="not connected"):
        store.import_documents([doc])
    with pytest.raises(ValueError, match="not connected"):
        store.get_backlinks(path_to_document_id("note.md"))
