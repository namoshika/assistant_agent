from unittest.mock import MagicMock

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters
from pytest_mock import MockerFixture

from assistant_agent.entities import postgres
from assistant_agent.loaders.obsidian import path_to_document_id
from assistant_agent.services.vault_obsidian import VaultObsidianRetriever
from assistant_agent.utils.store_factory import InMemoryStoreContext

_DOC_ID_A = path_to_document_id("a.md")
_DOC_ID_B = path_to_document_id("b.md")


@pytest.fixture
def retriever() -> VaultObsidianRetriever:
    """VaultObsidianRetriever のテスト用インスタンス.

    InMemoryStoreContext を注入し、外部依存なしで動作させる。
    """
    return VaultObsidianRetriever(
        sa_engine=MagicMock(),
        store_factory=InMemoryStoreContext(),
        docstore_name="test_docstore",
        vectorstore_name="test_vectorstore",
        embed_model=MagicMock(spec=BaseEmbedding),
        embed_dim=128,
        vault_entity=postgres.ObsidianEntity,
    )


def test_search_documents_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document を取得できるか確認  (filter 省略).

    観点1: as_retriever が引数 similarity_top_k 付きで呼ばれていること
    観点2: ref_doc_id が重複除去される (a.md は chunk1/chunk3 の 2 件あるが 1 件に集約)
    観点3: 初出順の deduplicated IDs が get_documents_by_ids に渡される

    TODO: 階層型レトリーバー導入の段階で試験内容を見直す
    """
    # 試験準備
    node_a1 = NodeWithScore(
        node=TextNode(relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=_DOC_ID_A)}),
        score=0.9,
    )
    node_b = NodeWithScore(
        node=TextNode(relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=_DOC_ID_B)}),
        score=0.8,
    )
    node_a2 = NodeWithScore(
        node=TextNode(relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=_DOC_ID_A)}),
        score=0.7,
    )

    m_llama_retriever = MagicMock()
    m_llama_retriever.retrieve.return_value = [node_a1, node_b, node_a2]
    m_index = MagicMock()
    m_index.as_retriever.return_value = m_llama_retriever
    mocker.patch(
        "assistant_agent.services.vault_obsidian.VectorStoreIndex.from_vector_store",
        return_value=m_index,
    )
    m_get_docs = mocker.patch.object(retriever, "get_documents_by_ids", return_value=[])

    # 試験実施
    retriever.search_documents("テスト", top_k=5, filters=None)

    # 結果検証
    # 観点1
    m_index.as_retriever.assert_called_once_with(similarity_top_k=5, filters=None)
    # 観点2 & 観点3: 重複除去かつ初出順の ID リストが渡される
    m_get_docs.assert_called_once_with([_DOC_ID_A, _DOC_ID_B])


def test_search_documents_02(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document を取得できるか確認 (filter 有り).

    観点1: as_retriever が filters 付きで呼ばれること
    """
    # 試験準備
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="date", value="2025-01-01 00:00:00", operator=FilterOperator.GTE)
        ]
    )

    m_llama_retriever = MagicMock()
    m_llama_retriever.retrieve.return_value = []

    m_index = MagicMock()
    m_index.as_retriever.return_value = m_llama_retriever
    mocker.patch(
        "assistant_agent.services.vault_obsidian.VectorStoreIndex.from_vector_store",
        return_value=m_index,
    )

    # 試験実施
    retriever.search_documents("テスト", top_k=5, filters=filters)

    # 結果検証
    # 観点1
    m_index.as_retriever.assert_called_once_with(similarity_top_k=5, filters=filters)


def test_search_documents_03(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document が無い場合に空リストを返せるか確認.

    観点1: retrieve() が空リストを返すとき get_documents_by_ids([]) が呼ばれ [] を返す
    """
    # 試験準備
    m_llama_retriever = MagicMock()
    m_llama_retriever.retrieve.return_value = []
    m_index = MagicMock()
    m_index.as_retriever.return_value = m_llama_retriever
    mocker.patch(
        "assistant_agent.services.vault_obsidian.VectorStoreIndex.from_vector_store",
        return_value=m_index,
    )
    m_get_docs = mocker.patch.object(retriever, "get_documents_by_ids", return_value=[])

    # 試験実施
    result = retriever.search_documents("クエリ", top_k=5, filters=None)

    # 結果検証
    # 観点1
    m_get_docs.assert_called_once_with([])
    assert result == []


def test_get_documents_by_ids_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """document_ids に対応する Document を取得できるか確認.

    観点1: DBが逆順で返しても document_ids の順序で返ること
    観点2: document_ids に存在しないIDが含まれる場合はスキップ
    観点3: document_ids が空リストのとき空リストを返す
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
    mocker.patch("assistant_agent.services.vault_obsidian.Session", return_value=m_session)

    # 観点1: DBが逆順 (b→a) で返しても document_ids の順 (a→b) で返る
    m_session.scalars.return_value.all.return_value = [m_row_b, m_row_a]
    docs = retriever.get_documents_by_ids([_DOC_ID_A, _DOC_ID_B])
    assert len(docs) == 2
    assert docs[0].text == "content_a"
    assert docs[1].text == "content_b"

    # 観点2: 存在しない ID はスキップ
    m_session.scalars.return_value.all.return_value = [m_row_a]
    unknown_id = "00000000-0000-0000-0000-000000000000"
    docs = retriever.get_documents_by_ids([_DOC_ID_A, unknown_id])
    assert len(docs) == 1
    assert docs[0].text == "content_a"

    # 観点3: 空リスト入力 → 空リストを返す
    m_session.scalars.return_value.all.return_value = []
    docs = retriever.get_documents_by_ids([])
    assert docs == []


def test_sync_chunks_01(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """Vault テーブルと Chunk テーブルの同期ができるか確認 (有件).

    観点1: 全 Document が LlamaDocument に変換され pipeline.run へ渡される
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
    mocker.patch("assistant_agent.services.vault_obsidian.Session", return_value=m_session)
    m_pipeline_run = mocker.patch("llama_index.core.ingestion.pipeline.IngestionPipeline.run")

    # 試験実施
    retriever.sync_chunks()

    # 結果検証
    # 観点1
    llama_docs = m_pipeline_run.call_args.kwargs["documents"]
    assert len(llama_docs) == 2
    assert llama_docs[0].doc_id == _DOC_ID_A
    assert llama_docs[0].text == "content_a"
    assert llama_docs[0].metadata == {"path": "a.md"}
    assert llama_docs[1].doc_id == _DOC_ID_B
    assert llama_docs[1].text == "content_b"
    assert llama_docs[1].metadata == {"path": "b.md"}


def test_sync_chunks_02(retriever: VaultObsidianRetriever, mocker: MockerFixture):
    """Vault テーブルと Chunk テーブルの同期ができるか確認 (0件).

    観点1: 空の Document リストが pipeline.run へ渡される
    """
    # 試験準備
    m_session = MagicMock()
    m_session.__enter__ = MagicMock(return_value=m_session)
    m_session.__exit__ = MagicMock(return_value=False)
    m_session.scalars.return_value.all.return_value = []
    mocker.patch("assistant_agent.services.vault_obsidian.Session", return_value=m_session)
    m_pipeline_run = mocker.patch("llama_index.core.ingestion.pipeline.IngestionPipeline.run")

    # 試験実施
    retriever.sync_chunks()

    # 結果検証
    # 観点1
    llama_docs = m_pipeline_run.call_args.kwargs["documents"]
    assert llama_docs == []
