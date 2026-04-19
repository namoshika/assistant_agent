from unittest.mock import MagicMock

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)
from pytest_mock import MockerFixture

from assistant_agent.entities import postgres
from assistant_agent.services import VaultSampleRetriever
from assistant_agent.utils.store_context import DuckDBStoreContext

_DOC_ID_A = "doc-id-a"
_DOC_ID_B = "doc-id-b"


@pytest.fixture
def retriever() -> VaultSampleRetriever:
    """VaultSampleRetriever のテスト用インスタンス.

    SA エンジンをモック化し、外部依存なしで動作させる。
    """
    return VaultSampleRetriever(
        docstore_name="test_docstore",
        vectorstore_name="test_vectorstore",
        store_context=DuckDBStoreContext(),
        transformations=[SentenceSplitter()],
        embed_model=MagicMock(spec=BaseEmbedding),
        embed_dim=128,
        vault_entity=postgres.SampleEntity,
    )


def test_search_documents_01(retriever: VaultSampleRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document を取得できるか確認  (filter 省略).

    観点1: as_retriever が引数 similarity_top_k 付きで呼ばれていること

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

    # 試験実施
    retriever.search_documents("テスト", top_k=5)

    # 結果検証
    # 観点1
    m_index.as_retriever.assert_called_once_with(similarity_top_k=5)
