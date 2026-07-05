from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pytest_mock import MockerFixture

from assistant_agent.entities import postgres
from assistant_agent.loaders.obsidian import path_to_document_id
from assistant_agent.services import VaultObsidianRetriever
from assistant_agent.services.vault_obsidian import DateFilter, SearchFilters

_DOC_ID_A = path_to_document_id("a.md")
_DOC_ID_B = path_to_document_id("b.md")


@pytest.fixture
def retriever() -> VaultObsidianRetriever:
    """VaultObsidianRetriever のテスト用インスタンス.

    store_conn は None を仮置きし、各テストで mocker.patch.object により差し込む。
    """
    return VaultObsidianRetriever(
        chunk_entity=postgres.ObsidianChunkEntity,
        store_conn=None,  # pyright: ignore[reportArgumentType]
        splitter=RecursiveCharacterTextSplitter(),
        embed_model=DeterministicFakeEmbedding(size=128),
        vault_entity=postgres.ObsidianEntity,
    )


def test_initialize_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """initialize() の遅延初期化動作を確認.

    観点1: initialize() を複数回呼んでもストア生成メソッドが1回しか呼ばれないこと
    観点2: search_documents の呼び出しが initialize() をトリガーすること（初回のみ）
    """
    # 試験準備
    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_store_conn.get_vector_store.return_value.similarity_search.return_value = []

    # 観点1: initialize() を2回呼んでもストア生成は1回
    retriever.initialize()
    retriever.initialize()
    m_store_conn.get_engine.assert_called_once()
    m_store_conn.get_vector_store.assert_called_once()

    # 観点2: search_documents 呼び出しで initialize() が再実行されないこと（呼び出し回数が増えない）
    retriever.search_documents("テスト", top_k=1, filters=None)
    m_store_conn.get_engine.assert_called_once()
    m_store_conn.get_vector_store.assert_called_once()


def test_search_documents_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document を取得できるか確認 (filter 省略).

    返却 Document は id + metadata のみ保持し page_content は空。

    観点1: similarity_search が引数 k 付き、filter=None で呼ばれていること
    観点2: document_id が重複除去される (a.md は chunk1/chunk3 の 2 件あるが 1 件に集約)
    観点3: 初出順で Document が返され、id と metadata が chunk から正しく引き継がれる
    """
    # 試験準備
    chunk_a1 = Document(
        id=_DOC_ID_A, page_content="a-1", metadata={"document_id": _DOC_ID_A, "file_path": "a.md"}
    )
    chunk_b1 = Document(
        id=_DOC_ID_B, page_content="b-1", metadata={"document_id": _DOC_ID_B, "file_path": "b.md"}
    )
    chunk_a2 = Document(
        id=_DOC_ID_A, page_content="a-2", metadata={"document_id": _DOC_ID_A, "file_path": "a.md"}
    )

    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_vector_store = m_store_conn.get_vector_store.return_value
    m_vector_store.similarity_search.return_value = [chunk_a1, chunk_b1, chunk_a2]

    # 試験実施
    result = retriever.search_documents("テスト", top_k=5, filters=None)

    # 結果検証
    # 観点1
    m_vector_store.similarity_search.assert_called_once_with("テスト", k=5, filter=None)
    # 観点2 & 観点3
    assert result[0] == chunk_a1.model_copy(update={"page_content": ""})
    assert result[1] == chunk_b1.model_copy(update={"page_content": ""})


def test_search_documents_02(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document リストを取得できるか確認 (filter 有り).

    観点1: similarity_search が filter に変換後の dict 付きで呼ばれること
    """
    # 試験準備
    filters = SearchFilters(date=DateFilter(gte="2025-01-01 00:00:00"))  # pyright: ignore[reportCallIssue]
    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_vector_store = m_store_conn.get_vector_store.return_value
    m_vector_store.similarity_search.return_value = []

    # 試験実施
    retriever.search_documents("テスト", top_k=5, filters=filters)

    # 結果検証
    # 観点1
    m_vector_store.similarity_search.assert_called_once_with(
        "テスト", k=5, filter={"date": {"$gte": "2025-01-01 00:00:00"}}
    )


def test_search_documents_03(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document が無い場合に空リストを返せるか確認.

    観点1: similarity_search() が空リストを返すとき [] を返す
    """
    # 試験準備
    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_store_conn.get_vector_store.return_value.similarity_search.return_value = []

    # 試験実施
    result = retriever.search_documents("クエリ", top_k=5, filters=None)

    # 結果検証
    # 観点1
    assert result == []


def test_get_documents_by_ids_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """document_ids に対応する Document を取得できるか確認.

    観点1: DBが逆順で返しても document_ids の順序で返ること
    観点2: document_ids に存在しないIDが含まれる場合はスキップ
    観点3: document_ids が空リストのとき空リストを返す
    """
    # 試験準備
    mocker.patch.object(retriever, "_store_conn")
    m_row_a, m_row_b = MagicMock(), MagicMock()
    m_row_a.document_id = _DOC_ID_A
    m_row_a.content = "content_a"
    m_row_a.document_metadata = {"file_path": "a.md"}
    m_row_b.document_id = _DOC_ID_B
    m_row_b.content = "content_b"
    m_row_b.document_metadata = {"file_path": "b.md"}

    m_session = MagicMock()
    m_session.__enter__ = MagicMock(return_value=m_session)
    m_session.__exit__ = MagicMock(return_value=False)
    mocker.patch("assistant_agent.services.vault_obsidian.Session", return_value=m_session)

    # 観点1: DBが逆順 (b→a) で返しても document_ids の順 (a→b) で返る
    m_session.scalars.return_value.all.return_value = [m_row_b, m_row_a]
    docs = retriever.get_documents_by_ids([_DOC_ID_A, _DOC_ID_B])
    assert len(docs) == 2
    assert docs[0].page_content == "content_a"
    assert docs[1].page_content == "content_b"

    # 観点2: 存在しない ID はスキップ
    m_session.scalars.return_value.all.return_value = [m_row_a]
    unknown_id = "00000000-0000-0000-0000-000000000000"
    docs = retriever.get_documents_by_ids([_DOC_ID_A, unknown_id])
    assert len(docs) == 1
    assert docs[0].page_content == "content_a"

    # 観点3: 空リスト入力 → 空リストを返す
    m_session.scalars.return_value.all.return_value = []
    docs = retriever.get_documents_by_ids([])
    assert docs == []


def test_sync_chunks_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """Vault テーブルと Chunk テーブルの同期ができるか確認 (有件).

    観点1: VaultUtils.sync_chunks が返す差分行が chunk 化され、add_documents に渡される。
        page_content が row.content から正しく伝播する
    観点2: forward_links はメタデータから除外される
    観点3: forward_links 以外のメタデータは型変換されずそのまま渡される (list を含む)
    観点4: 全チャンクに document_id/document_content_hash が伝播する
    """
    # 試験準備
    m_row_a, m_row_b = MagicMock(), MagicMock()
    m_row_a.document_id = _DOC_ID_A
    m_row_a.content = "content_a"
    m_row_a.document_content_hash = "hash_a"
    m_row_a.document_metadata = {
        "file_path": "a.md",
        "forward_links": ["id1", "id2"],
        "tags": ["tag1", "tag2"],
        "count": 3,
        "score": 1.5,
        "note": None,
    }
    m_row_b.document_id = _DOC_ID_B
    m_row_b.content = "content_b"
    m_row_b.document_content_hash = "hash_b"
    m_row_b.document_metadata = {"file_path": "b.md"}

    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_vector_store = m_store_conn.get_vector_store.return_value
    mocker.patch(
        "assistant_agent.services.vault_obsidian.base.VaultUtils.sync_chunks",
        return_value=[m_row_a, m_row_b],
    )

    # 試験実施
    retriever.sync_chunks()

    # 結果検証
    chunks = m_vector_store.add_documents.call_args.args[0]
    # 観点1
    assert len(chunks) == 2
    assert chunks[0].page_content == "content_a"
    assert chunks[1].page_content == "content_b"
    # 観点2
    assert "forward_links" not in chunks[0].metadata
    # 観点3
    assert chunks[0].metadata["tags"] == ["tag1", "tag2"]
    assert chunks[0].metadata["file_path"] == "a.md"
    assert chunks[0].metadata["count"] == 3
    assert chunks[0].metadata["score"] == 1.5
    assert chunks[0].metadata["note"] is None
    # 観点4
    assert chunks[0].metadata["document_id"] == _DOC_ID_A
    assert chunks[0].metadata["document_content_hash"] == "hash_a"
    assert chunks[1].metadata["document_id"] == _DOC_ID_B
    assert chunks[1].metadata["document_content_hash"] == "hash_b"


def test_sync_chunks_02(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """Vault テーブルと Chunk テーブルの同期ができるか確認 (差分なし).

    観点1: VaultUtils.sync_chunks が空リストを返すとき、add_documents も空リストで呼ばれる
    """
    # 試験準備
    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_vector_store = m_store_conn.get_vector_store.return_value
    mocker.patch(
        "assistant_agent.services.vault_obsidian.base.VaultUtils.sync_chunks",
        return_value=[],
    )

    # 試験実施
    retriever.sync_chunks()

    # 結果検証
    # 観点1
    chunks = m_vector_store.add_documents.call_args.args[0]
    assert chunks == []
