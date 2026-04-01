import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from sqlalchemy import Engine

from agent_assistant import connector, context
from agent_assistant.loader.obsidian import PgVault
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> "_FakeChatModel":
        return self


@pytest.mark.integration
def test_obsidian_vault_search_01(
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
) -> None:
    """obsidian_vault_search() を呼び出した時、 クエリと意味的に近いドキュメントを返せるか確認.

    観点1: sync 後に search が例外なく完了し、ToolMessage として返る
    観点2: ToolMessage.content が document_id の一覧を含む文字列で、
        artifact の各 Document が path メタデータを持つ
    """
    # 試験準備
    raw_entity = vault_entities
    PgVault.sync(vault_docs, sa_engine, raw_entity)
    obsidian_retriever.sync_chunks()

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {
                    "search_query": vault_docs[0].page_content[:20],
                    "filters": None,
                    "full_fetch": False,
                },
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, connector.obsidian_vault_search)

    # 試験実施
    result = agent.invoke(
        {"messages": [HumanMessage(content="ノートを検索して")]},
        context=context.ContextSchema(llm=MagicMock(), obsidian_store=obsidian_retriever),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    assert len(tool_msg.artifact) >= 1
    # 観点2
    assert len(tool_msg.content) > 0
    assert vault_docs[0].id is not None
    assert vault_docs[0].id in tool_msg.content
    for doc in tool_msg.artifact:
        assert "path" in doc.metadata


@pytest.mark.integration
def test_obsidian_vault_search_02() -> None:
    """実際の LLM から obsidian_vault_search をフィルタなしで呼び出せるか確認.

    観点1: LLM が search_query のみで tool call を生成し
        search_documents が filters=None で呼ばれる
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        pytest.fail("AWS 認証情報が未設定")

    # 試験準備
    docs = [Document(id="doc-1", page_content="本文", metadata={"path": "02_Daily/2026-01-01.md"})]
    m_store = MagicMock()
    m_store.search_documents.return_value = docs
    ctx = context.build_session()
    agent = create_agent(
        model=ctx.llm, tools=[connector.obsidian_vault_search], context_schema=context.ContextSchema
    )

    # 試験実施
    result = agent.invoke(
        {"messages": [HumanMessage(content="エージェントについてのノートを検索して")]},
        context=context.ContextSchema(llm=MagicMock(), obsidian_store=m_store),
    )

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    assert m_store.search_documents.call_args.kwargs.get("filters") is None


@pytest.mark.integration
def test_obsidian_vault_search_03() -> None:
    """実際の LLM から obsidian_vault_search をフィルタありで呼び出せるか確認.

    観点1: LLM が MetadataFilters を含む tool call を生成し
        search_documents が filters 付きで呼ばれる
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        pytest.fail("AWS 認証情報が未設定")

    # 試験準備
    docs = [Document(id="doc-1", page_content="本文", metadata={"path": "02_Daily/2026-01-01.md"})]
    m_store = MagicMock()
    m_store.search_documents.return_value = docs
    ctx = context.build_session()
    agent = create_agent(
        model=ctx.llm, tools=[connector.obsidian_vault_search], context_schema=context.ContextSchema
    )

    # 試験実施
    result = agent.invoke(
        {"messages": [HumanMessage(content="2026年1月以降のノートを検索して")]},
        context=context.ContextSchema(llm=MagicMock(), obsidian_store=m_store),
    )

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    filters = m_store.search_documents.call_args.kwargs.get("filters")
    assert filters is not None
    assert isinstance(filters, MetadataFilters)
    assert any(isinstance(f, MetadataFilter) and f.key == "date" for f in filters.filters)


@pytest.mark.integration
def test_obsidian_vault_get_01(
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
) -> None:
    """obsidian_vault_get() を呼び出した時、指定した document_id のドキュメントを取得できるか確認.

    観点1: document_id を指定するとノートを取得でき、ToolMessage として返る
    観点2: ToolMessage.content に page_content が含まれる
    観点3: 存在しない ID を指定すると ToolMessage.artifact が空リスト、content が空文字列
    """
    # 試験準備
    raw_entity = vault_entities
    PgVault.sync([vault_docs[0]], sa_engine, raw_entity)
    doc_id = vault_docs[0].id
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_get",
                "args": {"document_ids": [doc_id]},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, connector.obsidian_vault_get)

    # 試験実施
    result = agent.invoke(
        {"messages": [HumanMessage(content="ノートを取得して")]},
        context=context.ContextSchema(llm=MagicMock(), obsidian_store=obsidian_retriever),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert len(tool_msg.artifact) == 1
    # 観点2
    assert vault_docs[0].page_content[:10] in tool_msg.content
    ai_msg2 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_get",
                "args": {"document_ids": ["nonexistent.md"]},
                "id": "2",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg2, connector.obsidian_vault_get)
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="nonexistent.md を取得して")]},
        context=context.ContextSchema(llm=MagicMock(), obsidian_store=obsidian_retriever),
    )
    # 観点3
    tool_msg2 = next(m for m in result2["messages"] if isinstance(m, ToolMessage))
    assert tool_msg2.artifact == []
    assert tool_msg2.content == ""


@pytest.mark.integration
def test_format_documents_01(
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
) -> None:
    """Document リストを format_documents() に渡し、整形済み文字列を返せるか確認.

    観点: 戻り値が文字列である
    """
    # 試験準備
    raw_entity = vault_entities
    PgVault.sync([vault_docs[0]], sa_engine, raw_entity)
    docs = obsidian_retriever.get_documents_by_ids([vault_docs[0].id])

    # 試験実施
    result = connector.format_documents(docs, obsidian_retriever)

    # 結果検証
    assert isinstance(result, str)


def _make_agent(tool_calls_msg: AIMessage, *tools: Any) -> Any:
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools), context_schema=context.ContextSchema)
