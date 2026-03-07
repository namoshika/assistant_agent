# import os
# import sys
# from unittest.mock import MagicMock, patch

# import pytest
# from langchain_core.documents import Document
# from pytest_mock import MockerFixture

# # agent.py はトップレベルで外部接続を伴うオブジェクトを生成するため、
# # import 前に最小限のパッチを適用する
# os.environ.setdefault("ENV_PG_CONNECTION_STRING", "postgresql+asyncpg://u:p@localhost/db")
# os.environ.setdefault("ENV_GEMINI_API_KEY", "fake-key")

# _patches = [
#     patch("langchain_postgres.PGEngine.from_connection_string", return_value=MagicMock()),
#     patch("sqlalchemy.create_engine", return_value=MagicMock()),
#     patch("langchain.agents.create_agent", return_value=MagicMock()),
# ]
# for _p in _patches:
#     _p.start()
# sys.modules.pop("agent", None)
# import agent  # noqa: E402
# for _p in _patches:
#     _p.stop()


# def test_get_weather():
#     """固定の天気メッセージを返す。

#     観点: 引数の都市名が戻り値の文字列に含まれる
#     """
#     result = agent.get_weather.func("Tokyo")
#     assert "Tokyo" in result


# def test_search_knowledge_01(mocker: MockerFixture):
#     """月次ニュースを検索し、コンテンツ文字列と Document リストを返す。

#     観点1: doc_store.connect() が呼ばれる
#     観点2: doc_store.search_documents() が正しいクエリと top_k で呼ばれる
#     観点3: 戻り値のコンテンツ文字列に source と page_content が含まれる
#     観点4: 戻り値のアーティファクトが search_documents() の返り値と一致する
#     """
#     docs = [Document(page_content="ニュース本文", metadata={"source": "news.md"})]
#     m_store = MagicMock()
#     m_store.search_documents.return_value = docs
#     mocker.patch.object(agent, "doc_store", m_store)

#     content, results = agent.search_knowledge.func(search_query="AI")

#     # 観点1
#     m_store.connect.assert_called_once()
#     # 観点2
#     m_store.search_documents.assert_called_once_with("AI", top_k=5)
#     # 観点3
#     assert "news.md" in content
#     assert "ニュース本文" in content
#     # 観点4
#     assert results is docs


# def test_obsidian_vault_search_01(mocker: MockerFixture):
#     """Obsidian Vault をベクトル検索し、コンテンツ文字列と Document リストを返す。

#     観点1: obsidian_store.connect() が呼ばれる
#     観点2: obsidian_store.search_documents() が正しい引数で呼ばれる
#     観点3: 戻り値のコンテンツ文字列に path が title として含まれる
#     """
#     docs = [Document(page_content="ノート本文", metadata={"path": "notes/idea.md"})]
#     m_store = MagicMock()
#     m_store.search_documents.return_value = docs
#     mocker.patch.object(agent, "obsidian_store", m_store)

#     content, results = agent.obsidian_vault_search.func(search_query="アイデア")

#     # 観点1
#     m_store.connect.assert_called_once()
#     # 観点2
#     m_store.search_documents.assert_called_once_with("アイデア", top_k=5)
#     # 観点3
#     assert "notes/idea.md" in content
#     assert "ノート本文" in content


# def test_obsidian_vault_get_01(mocker: MockerFixture):
#     """ID 指定でノートを取得し、コンテンツ文字列と Document リストを返す。

#     観点1: obsidian_store.connect() が呼ばれる
#     観点2: obsidian_store.get_documents() に ids が渡される
#     観点3: 戻り値のコンテンツ文字列に path が title として含まれる
#     """
#     docs = [Document(page_content="取得したノート", metadata={"path": "folder/note.md"})]
#     m_store = MagicMock()
#     m_store.get_documents.return_value = docs
#     mocker.patch.object(agent, "obsidian_store", m_store)

#     content, results = agent.obsidian_vault_get.func(ids=["note.md"])

#     # 観点1
#     m_store.connect.assert_called_once()
#     # 観点2
#     m_store.get_documents.assert_called_once_with(["note.md"])
#     # 観点3
#     assert "folder/note.md" in content
#     assert "取得したノート" in content
