from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from assistant_agent.services.vault_databricks import VaultDatabricksRetriever

_INDEX_NAME = "catalog.schema.index"


@pytest.fixture
def retriever(mocker: MockerFixture) -> VaultDatabricksRetriever:
    """VaultDatabricksRetriever のテスト用インスタンス.

    DatabricksVectorSearch のインスタンス化をモック化し、外部依存なしで動作させる。
    """
    mocker.patch("assistant_agent.services.vault_databricks.DatabricksVectorSearch")
    return VaultDatabricksRetriever(index_name=_INDEX_NAME)


def test_search_documents_01(retriever: VaultDatabricksRetriever, mocker: MockerFixture):
    """search_documents が類似検索を実行し結果を返すこと.

    観点1: similarity_search が query=query, k=top_k で呼ばれること
    観点2: 戻り値が similarity_search の返却値と一致すること
    """
    # 試験準備
    expected = [Document(page_content="チャンクA"), Document(page_content="チャンクB")]
    m_similarity_search = MagicMock(return_value=expected)
    retriever._vector_store.similarity_search = m_similarity_search

    # 試験実施
    result = retriever.search_documents(query="テスト", top_k=3)

    # 結果検証
    # 観点1
    m_similarity_search.assert_called_once_with(query="テスト", k=3)
    # 観点2
    assert result == expected
