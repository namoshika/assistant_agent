import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pytest_mock import MockerFixture

from assistant_agent.entities import postgres
from assistant_agent.services import VaultSampleRetriever


@pytest.fixture
def retriever() -> VaultSampleRetriever:
    """VaultSampleRetriever のテスト用インスタンス.

    store_conn は None を仮置きし、各テストで mocker.patch.object により差し込む。
    """
    return VaultSampleRetriever(
        chunk_entity=postgres.SampleChunkEntity,
        store_conn=None,  # pyright: ignore[reportArgumentType]
        splitter=RecursiveCharacterTextSplitter(),
        embed_model=DeterministicFakeEmbedding(size=128),
        vault_entity=postgres.SampleEntity,
    )


def test_initialize_01(retriever: VaultSampleRetriever, mocker: MockerFixture):
    """initialize() の遅延初期化動作を確認.

    観点1: initialize() を複数回呼んでもストア生成メソッドが1回しか呼ばれないこと
    """
    # 試験準備
    m_store_conn = mocker.patch.object(retriever, "_store_conn")

    # 試験実施
    retriever.initialize()
    retriever.initialize()

    # 結果検証
    # 観点1
    m_store_conn.get_engine.assert_called_once()
    m_store_conn.get_vector_store.assert_called_once()


def test_search_documents_01(retriever: VaultSampleRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document を取得できるか確認.

    観点1: vector_store.similarity_search が引数 k 付きで呼ばれていること
    観点2: similarity_search の戻り値がそのまま返ること
    """
    # 試験準備
    docs = [Document(page_content="本文A"), Document(page_content="本文B")]
    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    m_vector_store = m_store_conn.get_vector_store.return_value
    m_vector_store.similarity_search.return_value = docs

    # 試験実施
    result = retriever.search_documents("テスト", top_k=5)

    # 結果検証
    # 観点1
    m_vector_store.similarity_search.assert_called_once_with("テスト", k=5)
    # 観点2
    assert result == docs
