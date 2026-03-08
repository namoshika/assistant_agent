import os
import uuid
import pytest
from langchain_core.documents import Document
from sqlalchemy import text
from pydantic import SecretStr

from agent_assistant.utils.chunker.text import TextChunker
from agent_assistant.retriever.obsidian import ObsidianChunkStore, ObsidianDocumentStore


@pytest.fixture()
def store_name():
    """テストごとに一意なテーブル名。"""
    return f"test_{uuid.uuid4().hex[:8]}_chunks"


@pytest.fixture()
def chunk_store(pg_engine, store_name: str):
    """テスト用 ObsidianChunkStore。ENV_GEMINI_API_KEY が必要。"""
    ENV_GEMINI_API_KEY = os.environ.get("ENV_GEMINI_API_KEY")
    if not ENV_GEMINI_API_KEY:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    ENV_GEMINI_API_KEY = SecretStr(ENV_GEMINI_API_KEY)
    yield ObsidianChunkStore(pg_engine, ENV_GEMINI_API_KEY)
    pg_engine.drop_table(store_name)


@pytest.fixture()
def obsidian_store(pg_engine, sa_engine):
    """実際の PostgreSQL に接続した ObsidianDocumentStore。

    ENV_GEMINI_API_KEY 環境変数が必要 (ObsidianChunkStore が os.getenv で自動取得)。
    テスト終了後に作成したテーブルを DROP する。
    """
    ENV_GEMINI_API_KEY = os.environ.get("ENV_GEMINI_API_KEY")
    if not ENV_GEMINI_API_KEY:
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")

    ENV_GEMINI_API_KEY = SecretStr(ENV_GEMINI_API_KEY)
    table_prefix = f"test_{uuid.uuid4().hex[:8]}"
    chunk_store = ObsidianChunkStore(pg_engine, ENV_GEMINI_API_KEY)
    store = ObsidianDocumentStore(
        store_name=table_prefix,
        chunk_store=chunk_store,
        sa_engine=sa_engine,
        chunker=TextChunker(chunk_size=128),
    )
    store.connect()
    yield store

    # teardown: テスト用テーブルを削除
    with sa_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_prefix}_raw CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS {table_prefix}_chunks CASCADE"))
        conn.commit()


@pytest.mark.integration
def test_get_vectorstore_01(chunk_store: ObsidianChunkStore, store_name: str):
    """実際の PostgreSQL に接続して VectorStore を取得できる。

    観点1: get_vectorstore() が例外なく VectorStore を返す
    観点2: 返り値が similarity_search メソッドを持つ
    """
    # 試験実施
    vs = chunk_store.get_vectorstore(store_name)

    # 結果検証
    # 観点1: 例外なく返る
    assert vs is not None
    # 観点2: VectorStore のインターフェースを持つ
    assert hasattr(vs, "similarity_search")


@pytest.mark.integration
def test_add_chunks_01(chunk_store: ObsidianChunkStore, store_name: str):
    """ドキュメントを追加後、similarity_search で取得できる。

    観点1: 1件追加後に similarity_search でドキュメントが取得できる
    観点2: path メタデータが保持されている
    観点3: 複数件追加後も similarity_search で取得できる
    """
    vs = chunk_store.get_vectorstore(store_name)

    # 1件追加
    chunk_store.add_chunks(
        store_name,
        [
            Document(
                page_content="PostgreSQL のベクターストア統合テスト",
                metadata={"path": "test.md", "start_index": 0},
            )
        ],
    )
    results = vs.similarity_search("PostgreSQL", k=1)
    assert len(results) >= 1
    assert "test.md" in [r.metadata.get("path") for r in results]

    # 複数件追加
    chunk_store.add_chunks(
        store_name,
        [
            Document(
                page_content=f"テスト文書 {i} の内容",
                metadata={"path": f"note_{i}.md", "start_index": 0},
            )
            for i in range(3)
        ],
    )
    results = vs.similarity_search("テスト文書", k=3)
    assert len(results) >= 1
    assert len({r.metadata.get("path") for r in results}) >= 1


@pytest.mark.integration
def test_import_documents_01(obsidian_store: ObsidianDocumentStore):
    """import_documents() が例外なく完了し、get_documents() で取得できる。

    観点1: import_documents() が例外を発生させない
    観点2: 挿入後に get_documents() でドキュメントを取得できる
    観点3: 返り値の page_content が登録時の内容と一致する
    観点4: 同一 path で2回 import すると upsert になる（重複なし・最新コンテンツ）
    """
    # 試験実施
    obsidian_store.import_documents(_make_docs())

    # 結果検証
    results = obsidian_store.get_documents(["langchain.md"])
    # 観点2: 1件返る
    assert len(results) == 1
    # 観点3: コンテンツが一致
    assert (
        results[0].page_content
        == "LangChain は LLM アプリケーション構築フレームワークである。"
    )

    # 観点4: upsert（同一 path で2回インポート）
    obsidian_store.import_documents(
        [
            Document(
                page_content="バージョン1の内容", metadata={"path": "upsert_test.md"}
            )
        ]
    )
    obsidian_store.import_documents(
        [
            Document(
                page_content="バージョン2の内容（更新後）",
                metadata={"path": "upsert_test.md"},
            )
        ]
    )
    results = obsidian_store.get_documents(["upsert_test.md"])
    assert len(results) == 1
    assert results[0].page_content == "バージョン2の内容（更新後）"


@pytest.mark.integration
def test_search_documents_01(obsidian_store: ObsidianDocumentStore):
    """import 後に search_documents() で意味的に近いドキュメントが返る。

    観点1: クエリに意味的に近い結果が1件以上返る
    観点2: 返り値の Document が path メタデータを持つ
    """
    # 試験準備
    obsidian_store.import_documents(_make_docs())

    # 試験実施
    results = obsidian_store.search_documents("LLM フレームワーク", top_k=2)

    # 結果検証
    # 観点1: 1件以上返る
    assert len(results) >= 1
    # 観点2: path メタデータを持つ
    for doc in results:
        assert "path" in doc.metadata


@pytest.mark.integration
def test_get_documents_01(obsidian_store: ObsidianDocumentStore):
    """ファイル名のみ（後方一致）で正しいドキュメントを取得できる。

    観点1: ファイル名のみの指定で 1 件返る
    観点2: 返り値の page_content が登録時の内容と一致する
    観点3: 存在しない path を指定すると空リストが返る
    """
    # 試験準備
    obsidian_store.import_documents(_make_docs())

    # 試験実施
    results = obsidian_store.get_documents(["postgres.md"])

    # 結果検証
    # 観点1: 1件返る
    assert len(results) == 1
    # 観点2: コンテンツに "PostgreSQL" が含まれる
    assert "PostgreSQL" in results[0].page_content
    # 観点3: 存在しない path → 空リスト
    assert obsidian_store.get_documents(["nonexistent_file.md"]) == []


@pytest.mark.integration
def test_get_backlinks_01(obsidian_store: ObsidianDocumentStore):
    """forward_links 付きドキュメントを import 後、get_backlinks() でバックリンク元を取得できる。

    観点1: note_a (forward_links: ["backlink_test_b.md"]) を import 後、
           get_backlinks(["backlink_test_b.md"]) で note_a が返る
    観点2: 返り値の Document が path メタデータを持つ
    観点3: 存在しない path を指定すると空リストが返る
    """
    # 試験準備
    docs = [
        Document(
            page_content="note_a の本文",
            metadata={
                "path": "backlink_test_a.md",
                "forward_links": ["backlink_test_b.md"],
            },
        ),
        Document(
            page_content="note_b の本文",
            metadata={"path": "backlink_test_b.md", "forward_links": []},
        ),
    ]
    obsidian_store.import_documents(docs)

    # 試験実施
    results = obsidian_store.get_backlinks(["backlink_test_b.md"])

    # 結果検証
    # 観点1: note_a が返る
    paths = [d.metadata["path"] for d in results]
    assert "backlink_test_a.md" in paths
    assert "backlink_test_b.md" not in paths
    # 観点2: path メタデータを持つ
    for doc in results:
        assert "path" in doc.metadata
    # 観点3: 存在しない path → 空リスト
    assert obsidian_store.get_backlinks(["nonexistent_target.md"]) == []


def _make_docs():
    """テスト用 Document リストを生成するヘルパー。"""
    return [
        Document(
            page_content="LangChain は LLM アプリケーション構築フレームワークである。",
            metadata={"path": "langchain.md", "tags": ["ai", "framework"]},
        ),
        Document(
            page_content="PostgreSQL は高性能なオープンソースデータベースである。",
            metadata={"path": "postgres.md", "tags": ["database"]},
        ),
    ]
