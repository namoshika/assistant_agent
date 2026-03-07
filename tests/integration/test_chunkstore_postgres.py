import os
import uuid
import pytest
from langchain_core.documents import Document

from agent_assistant.utils.retriever import ObsidianChunkStore


@pytest.fixture()
def store_name():
    """テストごとに一意なテーブル名。"""
    return f"test_{uuid.uuid4().hex[:8]}_chunks"


@pytest.fixture()
def chunk_store(pg_engine, store_name: str):
    """テスト用 ObsidianChunkStore。ENV_GEMINI_API_KEY が必要。"""
    if not os.environ.get("ENV_GEMINI_API_KEY"):
        pytest.fail("ENV_GEMINI_API_KEY が未設定のため失敗")
    yield ObsidianChunkStore(pg_engine)
    pg_engine.drop_table(store_name)


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
