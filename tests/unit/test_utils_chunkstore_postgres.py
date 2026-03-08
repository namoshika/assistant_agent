from pytest_mock import MockerFixture
from unittest.mock import MagicMock, call
from langchain_core.documents import Document
from langchain_postgres import Column

from agent_assistant.utils.chunkstore.postgres import PGVectorChunkStore


def test_get_vectorstore_01(mocker: MockerFixture):
    """テーブルが既存の場合、_init_table を呼ばずに VectorStore を返す。

    観点1: PGVectorStore.create_sync が期待引数で呼ばれる
    観点2: _init_table は呼ばれない
    観点3: 戻り値が create_sync の返り値と一致する
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "INT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        return_value=m_vectorstore,
    )

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
    )
    result = store.get_vectorstore("my_table")

    # 結果検証
    # 観点1: create_sync が正しい引数で呼ばれる
    m_create.assert_called_once_with(
        engine=engine,
        table_name="my_table",
        embedding_service=embedding,
        metadata_columns=[c.name for c in cols],
    )
    # 観点2: テーブル初期化は不要
    engine.init_vectorstore_table.assert_not_called()
    # 観点3: 戻り値が正しい
    assert result is m_vectorstore


def test_get_vectorstore_02(mocker: MockerFixture):
    """テーブル未存在で ValueError が発生した場合、テーブルを初期化して再取得する。

    観点1: create_sync が 2 回呼ばれる（1 回目は失敗、2 回目は成功）
    観点2: init_vectorstore_table が正しい引数で 1 回呼ばれる
    観点3: 最終的な戻り値が 2 回目の create_sync の返り値と一致する
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "INT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        side_effect=[ValueError("table not found"), m_vectorstore],
    )

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
    )
    result = store.get_vectorstore("new_table")

    # 観点1: create_sync が 2 回呼ばれる
    assert m_create.call_count == 2
    expected_call = call(
        engine=engine,
        table_name="new_table",
        embedding_service=embedding,
        metadata_columns=[c.name for c in cols],
    )
    m_create.assert_has_calls([expected_call, expected_call])
    # 観点2: テーブル初期化が正しい引数で呼ばれる
    engine.init_vectorstore_table.assert_called_once_with(
        "new_table",
        store.dimention_size,
        metadata_columns=cols,
    )
    # 観点3: 戻り値が正しい
    assert result is m_vectorstore


def test_add_chunks_01(mocker: MockerFixture):
    """add_chunks() が正しいテーブルの VectorStore にドキュメントを追加する。

    観点1: create_sync が store_name を引数に呼ばれる
    観点2: add_documents が正しいドキュメントリストで呼ばれる
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "INT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        return_value=m_vectorstore,
    )
    docs = [Document(page_content="text1"), Document(page_content="text2")]

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
    )
    store.add_chunks("my_table", docs)

    # 観点1: create_sync が store_name を引数に呼ばれる
    m_create.assert_called_once_with(
        engine=engine,
        table_name="my_table",
        embedding_service=embedding,
        metadata_columns=[c.name for c in cols],
    )
    # 観点2: add_documents にドキュメントが渡される
    m_vectorstore.add_documents.assert_called_once_with(docs)
