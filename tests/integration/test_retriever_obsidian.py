import os
import uuid
import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from agent_assistant.retriever.obsidian import (
    ObsidianChunkStore,
    ObsidianDocumentStore,
    path_to_document_id,
)
from tests.integration.conftest import make_docs


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


class TestObsidianChunkStore:
    @pytest.mark.integration
    def test_get_vectorstore_01(self, chunk_store: ObsidianChunkStore, store_name: str):
        """実際の PostgreSQL に接続して VectorStore を取得できる。

        観点1: get_vectorstore() が例外なく VectorStore を返す
        観点2: 返り値が similarity_search メソッドを持つ
        """
        # 試験実施
        vs = chunk_store.get_vectorstore(store_name)

        # 結果検証
        # 観点1
        assert vs is not None
        # 観点2
        assert hasattr(vs, "similarity_search")

    @pytest.mark.integration
    def test_add_chunks_01(self, chunk_store: ObsidianChunkStore, store_name: str):
        """ドキュメントを追加後、similarity_search で取得できる。

        観点1: 1件追加後に similarity_search でドキュメントが取得できる
        観点2: document_id メタデータが保持されている
        観点3: 複数件追加後も similarity_search で取得できる
        """
        # 試験準備
        vs = chunk_store.get_vectorstore(store_name)
        doc_id = path_to_document_id("test.md")

        # 試験実施
        chunk_store.add_chunks(
            store_name,
            [
                Document(
                    page_content="PostgreSQL のベクターストア統合テスト",
                    metadata={
                        "document_id": doc_id,
                        "path": "test.md",
                        "start_index": 0,
                    },
                )
            ],
        )

        # 結果検証
        results = vs.similarity_search("PostgreSQL", k=1)
        # 観点1
        assert len(results) >= 1
        # 観点2
        assert doc_id in [r.metadata.get("document_id") for r in results]

        # 観点3
        chunk_store.add_chunks(
            store_name,
            [
                Document(
                    page_content=f"テスト文書 {i} の内容",
                    metadata={
                        "document_id": path_to_document_id(f"note_{i}.md"),
                        "path": f"note_{i}.md",
                        "start_index": 0,
                    },
                )
                for i in range(3)
            ],
        )
        results = vs.similarity_search("テスト文書", k=3)
        assert len(results) >= 1
        assert len({r.metadata.get("document_id") for r in results}) >= 1


class TestObsidianDocumentStore:
    @pytest.mark.integration
    def test_connect_01(self, obsidian_store: ObsidianDocumentStore):
        """初回・2回目の connect() がいずれも正常終了する。

        観点1: 初回 connect() 後 _chunk_vs が非 None になる
        観点2: 2回目の connect() も例外を発生させない
        """
        # 試験実施 (1回目)
        obsidian_store.connect()

        # 結果検証 (1回目)
        # 観点1
        assert obsidian_store._chunk_vs is not None

        # 試験実施 (2回目)
        obsidian_store.connect()
        # 観点2: 例外なし (ここに到達すればOK)

    @pytest.mark.integration
    def test_import_documents_01(self, obsidian_store: ObsidianDocumentStore):
        """import_documents() が例外なく完了し、get_document_by_path() で取得できる。

        観点1: import_documents() が例外を発生させない
        観点2: 挿入後に get_document_by_path() でドキュメントを取得できる
        観点3: 返り値の page_content が登録時の内容と一致する
        観点4: 返り値の Document に document_id が含まれる
        観点5: 同一 path で2回 import すると upsert になる（重複なし・最新コンテンツ）
        """
        obsidian_store.connect()

        # 試験実施 (観点1)
        obsidian_store.import_documents(make_docs())

        # 結果検証
        results = obsidian_store.get_document_by_path("langchain.md")
        # 観点2
        assert len(results) == 1
        # 観点3
        assert (
            results[0].page_content
            == "LangChain は LLM アプリケーション構築フレームワークである。"
        )
        # 観点4
        assert "document_id" in results[0].metadata
        assert results[0].metadata["document_id"] == path_to_document_id("langchain.md")

        # 観点5
        obsidian_store.import_documents(
            [
                Document(
                    page_content="バージョン1の内容",
                    metadata={"path": "upsert_test.md"},
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
        results = obsidian_store.get_document_by_path("upsert_test.md")
        assert len(results) == 1
        assert results[0].page_content == "バージョン2の内容（更新後）"

    @pytest.mark.integration
    def test_search_documents_01(self, obsidian_store: ObsidianDocumentStore):
        """import 後に search_documents() で意味的に近いドキュメントが返る。

        観点1: クエリに意味的に近い結果が1件以上返る
        観点2: 返り値の Document が path メタデータを持つ
        観点3: 返り値の Document が document_id メタデータを持つ
        """
        obsidian_store.connect()

        # 試験準備
        obsidian_store.import_documents(make_docs())

        # 試験実施
        results = obsidian_store.search_documents("LLM フレームワーク", top_k=2)

        # 結果検証
        # 観点1
        assert len(results) >= 1
        for doc in results:
            # 観点2
            assert "path" in doc.metadata
            # 観点3
            assert "document_id" in doc.metadata

    @pytest.mark.integration
    def test_get_documents_by_ids_01(self, obsidian_store: ObsidianDocumentStore):
        """document_id の完全一致で正しいドキュメントを取得できる。

        観点1: 既存 document_id を指定すると 1 件返る
        観点2: 返り値の page_content が登録時の内容と一致する
        観点3: 存在しない document_id を指定すると空リストが返る
        """
        obsidian_store.connect()

        # 試験準備
        obsidian_store.import_documents(make_docs())
        doc_id = path_to_document_id("langchain.md")

        # 試験実施
        results = obsidian_store.get_documents_by_ids([doc_id])

        # 結果検証
        # 観点1
        assert len(results) == 1
        # 観点2
        assert (
            results[0].page_content
            == "LangChain は LLM アプリケーション構築フレームワークである。"
        )
        # 観点3
        assert obsidian_store.get_documents_by_ids(["nonexistent-uuid"]) == []

    @pytest.mark.integration
    def test_get_document_by_path_01(self, obsidian_store: ObsidianDocumentStore):
        """ファイル名のみ（後方一致）で正しいドキュメントを取得できる。

        観点1: ファイル名のみの指定で 1 件返る
        観点2: 返り値の page_content が登録時の内容と一致する
        観点3: 存在しない path を指定すると空リストが返る
        """
        obsidian_store.connect()

        # 試験準備
        obsidian_store.import_documents(make_docs())

        # 試験実施
        results = obsidian_store.get_document_by_path("postgres.md")

        # 結果検証
        # 観点1
        assert len(results) == 1
        # 観点2
        assert "PostgreSQL" in results[0].page_content
        # 観点3
        assert obsidian_store.get_document_by_path("nonexistent_file.md") == []

    @pytest.mark.integration
    def test_get_backlinks_01(self, obsidian_store: ObsidianDocumentStore):
        """forward_links 付きドキュメントを import 後、get_backlinks() でバックリンク元を取得できる。

        観点1: note_a (forward_links: [document_id of backlink_test_b]) を import 後、
               get_backlinks(document_id of backlink_test_b) で note_a が返る
        観点2: 返り値の Document が path メタデータを持つ
        観点3: 存在しない document_id を指定すると空リストが返る
        """
        obsidian_store.connect()

        # 試験準備
        doc_id_b = path_to_document_id("backlink_test_b.md")
        docs = [
            Document(
                page_content="note_a の本文",
                metadata={
                    "path": "backlink_test_a.md",
                    "forward_links": [doc_id_b],
                },
            ),
            Document(
                page_content="note_b の本文",
                metadata={"path": "backlink_test_b.md", "forward_links": []},
            ),
        ]
        obsidian_store.import_documents(docs)

        # 試験実施
        results = obsidian_store.get_backlinks(doc_id_b)

        # 結果検証
        # 観点1
        paths = [d.metadata["path"] for d in results]
        assert "backlink_test_a.md" in paths
        assert "backlink_test_b.md" not in paths
        # 観点2
        for doc in results:
            assert "path" in doc.metadata
        # 観点3
        assert (
            obsidian_store.get_backlinks(path_to_document_id("nonexistent_target.md"))
            == []
        )
