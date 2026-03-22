from unittest.mock import MagicMock, call

from langchain_core.documents import Document
from langchain_postgres import Column
from pytest_mock import MockerFixture
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from agent_assistant.utils.chunkstore.postgres import PGVectorChunkStore


class _TestBase(DeclarativeBase):
    pass


class _TestChunkEntity:
    key: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)
    page: Mapped[str] = mapped_column(String)


def test_get_vectorstore_01(mocker: MockerFixture):
    """テーブルが既存の場合、_init_table を呼ばずに VectorStore を返す.

    観点1: PGVectorStore.create_sync が期待引数で呼ばれる
    観点2: _init_table は呼ばれない
    観点3: 戻り値が create_sync の返り値と一致する
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "TEXT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        return_value=m_vectorstore,
    )

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        store_name="my_table",
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )
    result = store.get_vectorstore()

    # 結果検証
    # 観点1: create_sync が正しい引数で呼ばれる
    m_create.assert_called_once_with(
        engine=engine,
        embedding_service=embedding,
        table_name="my_table",
        metadata_columns=[c.name for c in cols],
    )
    # 観点2: テーブル初期化は不要
    engine.init_vectorstore_table.assert_not_called()
    # 観点3: 戻り値が正しい
    assert result is m_vectorstore


def test_get_vectorstore_02(mocker: MockerFixture):
    """テーブル未存在で ValueError が発生した場合、テーブルを初期化して再取得する.

    観点1: create_sync が 2 回呼ばれる（1 回目は失敗、2 回目は成功）
    観点2: init_vectorstore_table が正しい引数で 1 回呼ばれる
    観点3: 最終的な戻り値が 2 回目の create_sync の返り値と一致する
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "TEXT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        side_effect=[ValueError("table not found"), m_vectorstore],
    )

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        store_name="new_table",
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )
    result = store.get_vectorstore()

    # 観点1: create_sync が 2 回呼ばれる
    assert m_create.call_count == 2
    expected_call = call(
        engine=engine,
        embedding_service=embedding,
        table_name="new_table",
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
    """add_chunks() が store_name の VectorStore にドキュメントを追加する.

    観点1: create_sync が store_name を引数に呼ばれる
    観点2: add_documents が正しいドキュメントリストで呼ばれる
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "TEXT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        return_value=m_vectorstore,
    )
    docs = [Document(page_content="text1"), Document(page_content="text2")]

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        store_name="my_table",
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )
    store.add_chunks(docs)

    # 観点1: create_sync が store_name を引数に呼ばれる
    m_create.assert_called_once_with(
        engine=engine,
        table_name="my_table",
        embedding_service=embedding,
        metadata_columns=[c.name for c in cols],
    )
    # 観点2: add_documents にドキュメントが渡される
    m_vectorstore.add_documents.assert_called_once_with(docs)


def test_del_chunks_01(mocker: MockerFixture):
    """引数 chunk_ids を渡した場合、vectorstore.delete が chunk_ids と filter=None で呼ばれる.

    観点1: PGVectorStore.create_sync が store_name を引数に呼ばれる
    観点2: vectorstore.delete が chunk_ids と filter=None で呼ばれる
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "TEXT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        return_value=m_vectorstore,
    )
    chunk_ids = ["id1", "id2"]

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        store_name="my_table",
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )
    store.del_chunks(chunk_ids=chunk_ids)

    # 結果検証
    # 観点1: create_sync が store_name を引数に呼ばれる
    m_create.assert_called_once_with(
        engine=engine,
        table_name="my_table",
        embedding_service=embedding,
        metadata_columns=[c.name for c in cols],
    )
    # 観点2: delete に chunk_ids と filter=None が渡される
    m_vectorstore.delete.assert_called_once_with(chunk_ids, filter=None)


def test_del_chunks_02(mocker: MockerFixture):
    """引数 filter を渡した場合、vectorstore.delete が chunk_ids=None と filter で呼ばれる.

    観点1: PGVectorStore.create_sync が store_name を引数に呼ばれる
    観点2: vectorstore.delete が chunk_ids=None と filter で呼ばれる
    """
    # 試験準備
    engine = MagicMock()
    embedding = MagicMock()
    cols = [Column("source", "TEXT"), Column("page", "TEXT")]
    m_vectorstore = MagicMock()
    m_create = mocker.patch(
        "agent_assistant.utils.chunkstore.postgres.PGVectorStore.create_sync",
        return_value=m_vectorstore,
    )
    filter_ = {"source": "a"}

    # 試験実施
    store = PGVectorChunkStore(
        engine=engine,
        store_name="my_table",
        metadata_columns=cols,
        embedding=embedding,
        dimention_size=1536,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )
    store.del_chunks(filter=filter_)

    # 結果検証
    # 観点1: create_sync が store_name を引数に呼ばれる
    m_create.assert_called_once_with(
        engine=engine,
        table_name="my_table",
        embedding_service=embedding,
        metadata_columns=[c.name for c in cols],
    )
    # 観点2: delete に chunk_ids=None と filter が渡される
    m_vectorstore.delete.assert_called_once_with(None, filter=filter_)


def test_chunk_entity_01():
    """chunk_entity が store_name に対応する ORM マッピングクラスを返す.

    観点1: 返り値がクラス
    観点2: __tablename__ が store_name と一致する
    観点3: chunk_entity として渡したテーブル列の定義を持つ
    """
    # 試験準備
    store = PGVectorChunkStore(
        engine=MagicMock(),
        store_name="my_chunks",
        metadata_columns=[],
        embedding=MagicMock(),
        dimention_size=1536,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )

    # 試験実施 & 結果検証
    # 観点1
    assert isinstance(store.chunk_entity, type)
    # 観点2
    assert (
        store.chunk_entity.__tablename__  # pyright: ignore[reportAttributeAccessIssue]
        == "my_chunks"
    )
    # 観点3
    assert hasattr(store.chunk_entity, "key")
    assert hasattr(store.chunk_entity, "source")
    assert hasattr(store.chunk_entity, "page")
