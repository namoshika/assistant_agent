from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

import assistant_agent.tools.obsidian as tools
from assistant_agent import graph


class _FakeChatModel(GenericFakeChatModel):
    """bind_tools() をサポートするダミーチャットモデル."""

    def bind_tools(self, tools, **_):
        return self


def test_obsidian_vault_search_01():
    """Obsidian vault をベクトル検索し、ToolMessage として結果を返せるか確認.

    観点1: full_fetch=False のとき search_documents() が top_k=10 で呼ばれる
    観点2:
        結果が ToolMessage として返り、content に document_id が含まれ、
        artifact が search_documents() の返り値と一致する
    """
    # 試験準備
    docs = [
        Document(
            id="doc-id-1",
            page_content="ノート本文",
            metadata={"path": "notes/idea.md"},
        )
    ]
    m_store = MagicMock()
    m_store.search_documents.return_value = docs

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {"search_query": "アイデア", "filters": None, "full_fetch": False},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.obsidian_vault_search, context_schema=graph.ContextSchema)

    # 試験実施
    result = agent.invoke(
        {"messages": [HumanMessage(content="アイデアを検索して")]},
        context=graph.ContextSchema(llm=MagicMock(), obsidian_store=m_store),
    )

    # 結果検証
    # 観点1
    m_store.search_documents.assert_called_once_with("アイデア", top_k=10, filters=None)

    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "doc-id-1" in tool_msg.content
    assert "ノート本文" not in tool_msg.content
    assert tool_msg.artifact is docs


def test_obsidian_vault_search_02():
    """Obsidian vault をベクトル検索し、ToolMessage として結果を返せるか確認 (full_fetch=True).

    観点1: full_fetch=True のとき search_documents() が top_k=9999 で呼ばれる
    """
    # 試験準備
    m_store = MagicMock()
    m_store.search_documents.return_value = []

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {"search_query": "アイデア", "filters": None, "full_fetch": True},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.obsidian_vault_search, context_schema=graph.ContextSchema)

    # 試験実施
    agent.invoke(
        {"messages": [HumanMessage(content="全件取得して")]},
        context=graph.ContextSchema(llm=MagicMock(), obsidian_store=m_store),
    )

    # 結果検証
    # 観点1
    m_store.search_documents.assert_called_once_with("アイデア", top_k=9999, filters=None)


def test_obsidian_vault_search_03():
    """Obsidian vault をベクトル検索し、ToolMessage として結果を返せるか確認 (filters 有り).

    観点1: search_documents が filters 付きで呼ばれる
    """
    # 試験準備
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="date", value="2025-01-01 00:00:00", operator=FilterOperator.GTE)
        ]
    )
    docs = [
        Document(
            id="doc-id-1",
            page_content="ノート本文",
            metadata={"path": "notes/idea.md"},
        )
    ]
    m_store = MagicMock()
    m_store.search_documents.return_value = docs

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {
                    "search_query": "アイデア",
                    "filters": {
                        "filters": [
                            {"key": "date", "value": "2025-01-01 00:00:00", "operator": ">="}
                        ]
                    },
                    "full_fetch": False,
                },
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.obsidian_vault_search, context_schema=graph.ContextSchema)

    # 試験実施
    agent.invoke(
        {"messages": [HumanMessage(content="アイデアを検索して")]},
        context=graph.ContextSchema(llm=MagicMock(), obsidian_store=m_store),
    )

    # 結果検証
    # 観点1
    call_kwargs = m_store.search_documents.call_args
    assert call_kwargs.args == ("アイデア",)
    assert call_kwargs.kwargs["top_k"] == 10
    assert call_kwargs.kwargs["filters"] == filters


def test_obsidian_vault_get_01():
    """Obsidian vault からノートを document_id で取得し、ToolMessage として結果を返せるか確認.

    観点1: get_documents_by_ids() に ids リストが渡される
    観点2:
        結果が ToolMessage として返り、content に path と page_content が含まれ、artifact が
        get_documents_by_ids() の返り値と一致する
    """
    # 試験準備
    docs = [
        Document(
            id="doc-id-2",
            page_content="取得したノート",
            metadata={"path": "folder/note.md"},
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
    agent = _make_agent(ai_msg, tools.obsidian_vault_get, context_schema=graph.ContextSchema)

    # 試験実施
    result = agent.invoke(
        {"messages": [HumanMessage(content="note.md を取得して")]},
        context=graph.ContextSchema(llm=MagicMock(), obsidian_store=m_store),
    )

    # 結果検証
    # 観点1
    m_store.get_documents_by_ids.assert_called_once_with(["sample_document_id"])
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "title: note.md" in tool_msg.content
    assert "取得したノート" in tool_msg.content
    assert tool_msg.artifact is docs


def test_format_obs_docs_01():
    """Document リストから整形済み文字列を返せるか確認.

    観点: 戻り値が文字列である
    """
    # 試験準備
    docs = [
        Document(
            id="doc-id-1",
            page_content="ノート本文",
            metadata={"path": "folder/note.md", "tags": "日本語テキスト"},
        )
    ]
    m_store = MagicMock()

    # 試験実施
    result = tools.format_docs_obs(docs, m_store)

    # 結果検証
    assert isinstance(result, str)


def test_format_document_ids_01():
    """Document リストから document_id と path の一覧文字列を返せるか確認.

    観点1: 戻り値が文字列である
    観点2: document_id と path が含まれ、page_content は含まれない
    """
    # 試験準備
    docs = [
        Document(
            id="doc-id-1",
            page_content="ノート本文",
            metadata={"path": "notes/idea.md"},
        ),
        Document(
            id="doc-id-2",
            page_content="別のノート",
            metadata={"path": "folder/meeting.md"},
        ),
    ]

    # 試験実施
    result = tools.format_document_ids(docs)

    # 結果検証
    # 観点1
    assert isinstance(result, str)
    # 観点2
    assert "doc-id-1" in result
    assert "doc-id-2" in result
    assert "ノート本文" not in result
    assert "別のノート" not in result


def _make_agent(tool_calls_msg: AIMessage, *tools, context_schema=None):
    """テスト用エージェントを生成するヘルパー."""
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools), context_schema=context_schema)
