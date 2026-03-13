import os
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent

from agent_assistant import connector
from agent_assistant.context import ContextSchema


class _FakeChatModel(GenericFakeChatModel):
    """bind_tools() をサポートするダミーチャットモデル。"""

    def bind_tools(self, tools, **_):
        return self


def _make_agent(tool_calls_msg: AIMessage, *tools, context_schema=None):
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(
        model=fake_llm, tools=list(tools), context_schema=context_schema
    )


def test_get_llm_01():
    """get_llm() が BaseChatModel を返す。

    観点: 戻り値が BaseChatModel のインスタンスである
    """
    # 試験実施
    llm = connector.get_llm()

    # 結果検証
    # 観点: 戻り値が BaseChatModel のインスタンスである
    assert isinstance(llm, BaseChatModel)


def test_get_weather_01():
    """天気を取得する。

    観点: ToolMessage として返り、content に引数の都市名が含まれる
    """
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"city": "Tokyo"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, connector.get_weather)
    result = agent.invoke({"messages": [HumanMessage(content="東京の天気は?")]})

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Tokyo" in tool_msg.content


def test_obsidian_vault_search_01():
    """Obsidian vault をベクトル検索し、ToolMessage として結果を返す。

    観点1: obsidian_store.connect() が呼ばれる
    観点2: search_documents() が正しいクエリと top_k=5 で呼ばれる
    観点3: 結果が ToolMessage として返り、content に path と page_content が含まれ、artifact が search_documents() の返り値と一致する
    """
    docs = [
        Document(
            page_content="ノート本文",
            metadata={"path": "notes/idea.md", "document_id": "doc-id-1"},
        )
    ]
    m_store = MagicMock()
    m_store.search_documents.return_value = docs

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {"search_query": "アイデア"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(
        ai_msg, connector.obsidian_vault_search, context_schema=ContextSchema
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="アイデアを検索して")]},
        context=ContextSchema(obsidian_store=m_store),
    )

    # 観点1
    m_store.connect.assert_called_once()
    # 観点2
    m_store.search_documents.assert_called_once_with("アイデア", top_k=5)
    # 観点3
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "title: idea.md" in tool_msg.content
    assert "ノート本文" in tool_msg.content
    assert tool_msg.artifact is docs


def test_obsidian_vault_get_01():
    """ID 指定でノートを取得し、ToolMessage として結果を返す。

    観点1: obsidian_store.connect() が呼ばれる
    観点2: get_documents() に ids リストが渡される
    観点3: 結果が ToolMessage として返り、content に path と page_content が含まれ、artifact が get_documents() の返り値と一致する
    """
    docs = [
        Document(
            page_content="取得したノート",
            metadata={"path": "folder/note.md", "document_id": "doc-id-2"},
        )
    ]
    m_store = MagicMock()
    m_store.get_documents_by_ids.return_value = docs

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_get",
                "args": {"document_ids": ["sample_document_id"]},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(
        ai_msg, connector.obsidian_vault_get, context_schema=ContextSchema
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="note.md を取得して")]},
        context=ContextSchema(obsidian_store=m_store),
    )

    # 観点1
    m_store.connect.assert_called_once()
    # 観点2
    m_store.get_documents_by_ids.assert_called_once_with(["sample_document_id"])
    # 観点3
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "title: note.md" in tool_msg.content
    assert "取得したノート" in tool_msg.content
    assert tool_msg.artifact is docs


def test_format_documents_01():
    """document リストから整形済み文字列を返す。

    観点: 戻り値が文字列である
    """
    # 試験準備
    docs = [
        Document(
            page_content="ノート本文",
            metadata={"path": "folder/note.md", "document_id": "doc-id-1", "tags": "日本語テキスト"},
        )
    ]
    m_store = MagicMock()

    # 試験実施
    result = connector.format_documents(docs, m_store)

    # 結果検証
    assert isinstance(result, str)
