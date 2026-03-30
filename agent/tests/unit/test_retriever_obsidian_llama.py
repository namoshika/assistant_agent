from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from agent_assistant.loader.obsidian import path_to_document_id
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever

_DOC_ID_A = path_to_document_id("a.md")
_DOC_ID_B = path_to_document_id("b.md")


@pytest.fixture
def retriever(mocker: MockerFixture) -> ObsidianLlamaRetriever:
    """ObsidianLlamaRetriever のテスト用インスタンス.

    LlamaIndex の外部依存 (PGVectorStore, PostgresDocumentStore, IngestionPipeline) をモック化する。
    """
    mocker.patch("agent_assistant.retriever.obsidian_llama.PGVectorStore")
    mocker.patch("agent_assistant.retriever.obsidian_llama.PostgresDocumentStore")
    mocker.patch("agent_assistant.retriever.obsidian_llama.IngestionPipeline")
    return ObsidianLlamaRetriever(
        sa_engine=MagicMock(),
        connection_string="postgresql://test",
        docstore_name="test_docstore",
        vectorstore_name="test_vectorstore",
        embed_model=MagicMock(),
        embed_dim=128,
    )


def test_search_documents_01(retriever: ObsidianLlamaRetriever, mocker: MockerFixture):
    """Chunk 検索 → ref_doc_id 重複除去 → raw から全文取得 → 初出順ソート.

    観点1: as_retriever が top_k を引数に呼ばれる
    観点2: ref_doc_id が重複除去される (a.md は chunk1/chunk3 の 2 件あるが 1 件に集約)
    観点3: ORM で全文取得が行われ Document リストが返る
    観点4: 類似検索の初出順が保持される (DB が逆順で返しても正しく並ぶ)

    TODO: 階層型レトリーバー導入の段階で試験内容を見直す
    """
    # 試験準備
    node_a1, node_b, node_a2 = MagicMock(), MagicMock(), MagicMock()
    node_a1.node.ref_doc_id = _DOC_ID_A
    node_b.node.ref_doc_id = _DOC_ID_B
    node_a2.node.ref_doc_id = _DOC_ID_A

    m_llama_retriever = MagicMock()
    m_llama_retriever.retrieve.return_value = [node_a1, node_b, node_a2]

    m_index = MagicMock()
    m_index.as_retriever.return_value = m_llama_retriever
    mocker.patch(
        "agent_assistant.retriever.obsidian_llama.VectorStoreIndex.from_vector_store",
        return_value=m_index,
    )

    m_row_a, m_row_b = MagicMock(), MagicMock()
    m_row_a.document_id = _DOC_ID_A
    m_row_a.content = "content_a"
    m_row_a.document_metadata = {"path": "a.md"}
    m_row_b.document_id = _DOC_ID_B
    m_row_b.content = "content_b"
    m_row_b.document_metadata = {"path": "b.md"}

    m_session = MagicMock()
    m_session.__enter__ = MagicMock(return_value=m_session)
    m_session.__exit__ = MagicMock(return_value=False)
    m_session.scalars.return_value.all.return_value = [m_row_b, m_row_a]
    mocker.patch("agent_assistant.retriever.obsidian_llama.Session", return_value=m_session)

    # 試験実施
    docs = retriever.search_documents("テスト", top_k=5)

    # 結果検証
    # 観点1
    m_index.as_retriever.assert_called_once_with(similarity_top_k=5)
    # 観点2 & 3
    assert len(docs) == 2
    # 観点4: DB が逆順 (b→a) でも初出順 (a→b) で返る
    assert docs[0].page_content == "content_a"
    assert docs[1].page_content == "content_b"


def test_search_documents_02(retriever: ObsidianLlamaRetriever, mocker: MockerFixture):
    """retrieve() が空リストを返すとき search_documents は [] を返す.

    観点1: 空リストが返る
    TODO: 階層型レトリーバー導入の段階で試験内容を見直す
    """
    # 試験準備
    m_llama_retriever = MagicMock()
    m_llama_retriever.retrieve.return_value = []

    m_index = MagicMock()
    m_index.as_retriever.return_value = m_llama_retriever
    mocker.patch(
        "agent_assistant.retriever.obsidian_llama.VectorStoreIndex.from_vector_store",
        return_value=m_index,
    )

    # 試験実施
    docs = retriever.search_documents("クエリ", top_k=5)

    # 結果検証
    # 観点1
    assert docs == []


def test_sync_chunks_01(retriever: ObsidianLlamaRetriever, mocker: MockerFixture):
    """Vault テーブルとチャンクテーブルの同期ができること (有件).

    観点1: 全ドキュメントが LlamaDocument に変換され pipeline.run へ渡される
    観点2: sync_chunks 後に _index キャッシュが None にリセットされる
    """
    # 試験準備
    m_row_a, m_row_b = MagicMock(), MagicMock()
    m_row_a.document_id = _DOC_ID_A
    m_row_a.content = "content_a"
    m_row_a.document_metadata = {"path": "a.md"}
    m_row_b.document_id = _DOC_ID_B
    m_row_b.content = "content_b"
    m_row_b.document_metadata = {"path": "b.md"}

    m_session = MagicMock()
    m_session.__enter__ = MagicMock(return_value=m_session)
    m_session.__exit__ = MagicMock(return_value=False)
    m_session.scalars.return_value.all.return_value = [m_row_a, m_row_b]
    mocker.patch("agent_assistant.retriever.obsidian_llama.Session", return_value=m_session)

    retriever._index = MagicMock()

    # 試験実施
    retriever.sync_chunks()

    # 結果検証
    # 観点1
    pipeline_run: MagicMock = retriever._pipeline.run  # type: ignore[assignment]
    llama_docs = pipeline_run.call_args.kwargs["documents"]
    assert len(llama_docs) == 2
    assert llama_docs[0].doc_id == _DOC_ID_A
    assert llama_docs[0].text == "content_a"
    assert llama_docs[0].metadata == {"path": "a.md"}
    assert llama_docs[1].doc_id == _DOC_ID_B
    assert llama_docs[1].text == "content_b"
    assert llama_docs[1].metadata == {"path": "b.md"}
    # 観点2
    assert retriever._index is None


def test_sync_chunks_02(retriever: ObsidianLlamaRetriever, mocker: MockerFixture):
    """Vault テーブルとチャンクテーブルの同期ができること (0件).

    観点1: 空のドキュメントリストが pipeline.run へ渡される
    """
    # 試験準備
    m_session = MagicMock()
    m_session.__enter__ = MagicMock(return_value=m_session)
    m_session.__exit__ = MagicMock(return_value=False)
    m_session.scalars.return_value.all.return_value = []
    mocker.patch("agent_assistant.retriever.obsidian_llama.Session", return_value=m_session)

    # 試験実施
    retriever.sync_chunks()

    # 結果検証
    # 観点1
    pipeline_run: MagicMock = retriever._pipeline.run  # type: ignore[assignment]
    llama_docs = pipeline_run.call_args.kwargs["documents"]
    assert llama_docs == []
