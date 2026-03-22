import os
from collections.abc import Generator

import pytest
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import Column, PGEngine
from pydantic import SecretStr
from sqlalchemy import Engine, MetaData, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from agent_assistant.loader.obsidian import path_to_document_id
from agent_assistant.utils.chunkstore.postgres import PGVectorChunkStore


class _TestBase(DeclarativeBase):
    metadata = MetaData("public")


class _TestChunkEntity:
    key: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)


@pytest.fixture()
def chunk_store(
    pg_engine: PGEngine, sa_engine: Engine, vault_name: str
) -> Generator[PGVectorChunkStore, None, None]:
    """実際の PostgreSQL に接続した PGVectorChunkStore.

    ENV_GEMINI_API_KEY 環境変数が必要。テスト終了後に作成したテーブルを DROP する。
    """
    env_gemini_api_key = os.environ.get("ENV_GEMINI_API_KEY")
    if not env_gemini_api_key:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    store = PGVectorChunkStore(
        engine=pg_engine,
        store_name=f"{vault_name}_chunks",
        metadata_columns=[Column("source", "text", False)],
        embedding=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", api_key=SecretStr(env_gemini_api_key)
        ),
        dimention_size=3072,
        chunk_entity=_TestChunkEntity,
        chunk_base=_TestBase,
    )
    yield store

    with sa_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {vault_name}_chunks CASCADE"))
        conn.commit()


@pytest.mark.integration
def test_add_chunks_01(chunk_store: PGVectorChunkStore):
    """add_chunks でドキュメントが追加される.

    観点1: 追加したドキュメントが similarity_search で取得できる
    """
    # 試験準備
    docs = [
        Document(
            page_content="add_chunks 統合テスト用ドキュメント",
            metadata={"source": "add_test.md"},
        )
    ]

    # 試験実施
    chunk_store.add_chunks(docs)

    # 結果検証
    vs = chunk_store.get_vectorstore()
    # 観点1: 追加したドキュメントが検索で返る
    results = vs.similarity_search(
        "add_chunks 統合テスト用ドキュメント",
        k=1,
        filter={"source": "add_test.md"},
    )
    assert len(results) == 1


@pytest.mark.integration
def test_del_chunks_01(chunk_store: PGVectorChunkStore):
    """chunk_ids と filter それぞれでチャンクが削除される.

    観点1: chunk_ids 指定で対象ドキュメントが削除される
    観点2: filter 指定で同一メタデータを持つ複数ドキュメントが一括削除される
    観点3: filter 削除で対象外のドキュメントは残る
    """
    # 試験準備
    doc_id = path_to_document_id("source_a.md")
    chunk_store.add_chunks(
        [
            Document(
                id=doc_id,
                page_content="chunk_ids 削除テスト用ドキュメント",
                metadata={"source": "source_a.md"},
            ),
            Document(
                page_content="filter 削除テスト A-1",
                metadata={"source": "group_a.md"},
            ),
            Document(
                page_content="filter 削除テスト A-2",
                metadata={"source": "group_a.md"},
            ),
            Document(
                page_content="filter 削除テスト B-1",
                metadata={"source": "group_b.md"},
            ),
            Document(
                page_content="filter 削除テスト C-1",
                metadata={"source": "group_c.md"},
            ),
        ]
    )

    # 試験実施: chunk_ids で削除
    chunk_store.del_chunks(chunk_ids=[doc_id])
    # 試験実施: $in フィルタで複数 source を一括削除
    chunk_store.del_chunks(filter={"source": {"$in": ["group_a.md", "group_b.md"]}})

    # 結果検証
    vs = chunk_store.get_vectorstore()
    # 観点1: chunk_ids 指定の削除対象が消えている
    assert (
        vs.similarity_search("chunk_ids 削除テスト", k=10, filter={"source": "source_a.md"}) == []
    )
    # 観点2: $in 指定の削除対象 (group_a 2件・group_b 1件) が消えている
    assert vs.similarity_search("filter 削除テスト A", k=10, filter={"source": "group_a.md"}) == []
    assert vs.similarity_search("filter 削除テスト B", k=10, filter={"source": "group_b.md"}) == []
    # 観点3: $in 対象外の group_c は残っている
    assert (
        len(vs.similarity_search("filter 削除テスト C", k=10, filter={"source": "group_c.md"})) == 1
    )
