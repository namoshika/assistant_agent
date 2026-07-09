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


async def test_initialize_01(retriever: VaultSampleRetriever, mocker: MockerFixture):
    """initialize() の遅延初期化動作を確認.

    観点1: initialize() を複数回呼んでもストア生成メソッドが1回しか呼ばれないこと
    """
    # 試験準備
    m_store_conn = mocker.patch.object(retriever, "_store_conn")
    mocker.patch("assistant_agent.services.vault_sample.PGEngine.from_engine")
    m_create = mocker.patch(
        "assistant_agent.services.vault_sample.PGVectorStore.create",
        new_callable=mocker.AsyncMock,
    )

    # 試験実施
    await retriever.initialize()
    await retriever.initialize()

    # 結果検証
    # 観点1
    m_store_conn.get_engine.assert_called_once()
    m_create.assert_called_once()


async def test_search_documents_01(retriever: VaultSampleRetriever, mocker: MockerFixture):
    """クエリで Chunk 検索し、類似する Document を取得できるか確認.

    観点1: vector_store.asimilarity_search が引数 k 付きで呼ばれていること
    観点2: asimilarity_search の戻り値がそのまま返ること
    """
    # 試験準備
    docs = [Document(page_content="本文A"), Document(page_content="本文B")]
    mocker.patch.object(retriever, "_store_conn")
    mocker.patch("assistant_agent.services.vault_sample.PGEngine.from_engine")
    m_create = mocker.patch(
        "assistant_agent.services.vault_sample.PGVectorStore.create",
        new_callable=mocker.AsyncMock,
    )
    m_vector_store = m_create.return_value
    m_vector_store.asimilarity_search = mocker.AsyncMock(return_value=docs)

    # 試験実施
    result = await retriever.search_documents("テスト", top_k=5)

    # 結果検証
    # 観点1
    m_vector_store.asimilarity_search.assert_called_once_with("テスト", k=5)
    # 観点2
    assert result == docs
