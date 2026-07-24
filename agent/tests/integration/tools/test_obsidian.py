from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import assistant_agent.tools.obsidian as tools
from assistant_agent.entities.base import VaultUtils
from assistant_agent.services import VaultObsidianRetriever
from assistant_agent.services.vault_obsidian import SearchFilters
from assistant_agent.store import PostgresStoreConnector
from assistant_agent.utils.context import CommonContext


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> _FakeChatModel:
        return self


@pytest.mark.integration
async def test_obsidian_vault_search_01(
    pg_conn: PostgresStoreConnector,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type,
    docs_obs: list[Document],
) -> None:
    """obsidian_vault_search() を呼び出した時、 クエリと意味的に近いドキュメントを返せるか確認.

    観点1: sync 後に search が例外なく完了し、ToolMessage として返る
    観点2: ToolMessage.content が document_id の一覧を含む文字列で、
        artifact の各 Document が path メタデータを持つ
    """
    # 試験準備
    raw_entity = pg_entity_obs
    await VaultUtils.sync_docs(docs_obs, pg_conn.get_engine(), raw_entity)
    await pg_retriever_obs.sync_chunks()

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {
                    "search_query": docs_obs[0].page_content[:20],
                    "filters": None,
                    "full_fetch": False,
                },
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.obsidian_vault_search)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="ノートを検索して")]},
        context={"obsidian_retriever": pg_retriever_obs},
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    assert len(tool_msg.artifact) >= 1
    # 観点2
    assert len(tool_msg.content) > 0
    assert docs_obs[0].id is not None
    assert docs_obs[0].id in tool_msg.content
    for doc in tool_msg.artifact:
        assert "file_path" in doc.metadata


@pytest.mark.integration
async def test_obsidian_vault_search_02(llm: BaseChatModel) -> None:
    """実際の LLM から obsidian_vault_search をフィルタなしで呼び出せるか確認.

    観点1: LLM が search_query のみで tool call を生成し
        search_documents が filters=None で呼ばれる
    """
    # 試験準備
    docs = [
        Document(id="doc-1", page_content="本文", metadata={"file_path": "02_Daily/2026-01-01.md"})
    ]
    m_store = MagicMock()
    m_store.search_documents = AsyncMock(return_value=docs)
    agent = create_agent(
        model=llm, tools=[tools.obsidian_vault_search], context_schema=CommonContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="エージェントのノートを1回検索し document_id を出す")]},
        context=cast(CommonContext, {"obsidian_retriever": m_store}),
    )

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    assert m_store.search_documents.call_args.kwargs.get("filters") is None


@pytest.mark.integration
async def test_obsidian_vault_search_03(llm: BaseChatModel) -> None:
    """実際の LLM から obsidian_vault_search をフィルタありで呼び出せるか確認.

    観点1: LLM が MetadataFilters を含む tool call を生成し
        search_documents が filters 付きで呼ばれる
    """
    # 試験準備
    docs = [
        Document(id="doc-1", page_content="本文", metadata={"file_path": "02_Daily/2026-01-01.md"})
    ]
    m_store = MagicMock()
    m_store.search_documents = AsyncMock(return_value=docs)
    agent = create_agent(
        model=llm, tools=[tools.obsidian_vault_search], context_schema=CommonContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="2026/01 以降のノートを検索し、 document_id を出して")]},
        context=cast(CommonContext, {"obsidian_retriever": m_store}),
    )

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    filters = m_store.search_documents.call_args.kwargs.get("filters")
    assert filters is not None
    assert isinstance(filters, SearchFilters)
    assert filters.date is not None


@pytest.mark.integration
async def test_obsidian_vault_get_01(
    pg_conn: PostgresStoreConnector,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type,
    docs_obs: list[Document],
) -> None:
    """obsidian_vault_get() を呼び出した時、指定した document_id のドキュメントを取得できるか確認.

    観点1: document_id を指定するとノートを取得でき、ToolMessage として返る
    観点2: ToolMessage.content に page_content が含まれる
    観点3: 存在しない ID を指定すると ToolMessage.artifact が空リスト、content が空文字列
    """
    # 試験準備
    raw_entity = pg_entity_obs
    await VaultUtils.sync_docs([docs_obs[0]], pg_conn.get_engine(), raw_entity)
    doc_id = docs_obs[0].id
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
    agent = _make_agent(ai_msg, tools.obsidian_vault_get)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="ノートを取得して")]},
        context=cast(CommonContext, {"obsidian_retriever": pg_retriever_obs}),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert len(tool_msg.artifact) == 1
    # 観点2
    assert docs_obs[0].page_content[:10] in tool_msg.content
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
    agent = _make_agent(ai_msg2, tools.obsidian_vault_get)
    result2 = await agent.ainvoke(
        {"messages": [HumanMessage(content="nonexistent.md を取得して")]},
        context=cast(CommonContext, {"obsidian_retriever": pg_retriever_obs}),
    )
    # 観点3
    tool_msg2 = next(m for m in result2["messages"] if isinstance(m, ToolMessage))
    assert tool_msg2.artifact == []
    assert tool_msg2.content == ""


def _make_agent(tool_calls_msg: AIMessage, *tools: Any) -> Any:
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools))
