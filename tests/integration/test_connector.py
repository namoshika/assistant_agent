import os
import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent
from sqlalchemy import Engine
from typing import Any

from agent_assistant import connector
from agent_assistant.context import ContextSchema
from agent_assistant.loader.obsidian import PgVault
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **_: Any) -> "_FakeChatModel":
        return self


def _make_agent(tool_calls_msg: AIMessage, *tools: Any) -> Any:
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools), context_schema=ContextSchema)


@pytest.mark.integration
def test_format_documents_01(
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
) -> None:
    """document リストを format_documents() に渡し、整形済み文字列を返せる。

    観点: 戻り値が文字列である
    """
    # 試験準備
    raw_entity = vault_entities
    PgVault.sync([vault_docs[0]], sa_engine, raw_entity)
    docs = obsidian_retriever.get_documents_by_ids(
        [vault_docs[0].metadata["document_id"]]
    )

    # 試験実施
    result = connector.format_documents(docs, obsidian_retriever)

    # 結果検証
    assert isinstance(result, str)


@pytest.mark.integration
def test_get_llm_01() -> None:
    """get_llm() を呼び出した時、正常動作する BaseChatModel インスタンスを返せる。

    観点1: 戻り値が BaseChatModel のインスタンスである
    観点2: invoke() を呼ぶと AIMessage としてレスポンスが返る
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        pytest.fail("AWS 認証情報が未設定")

    # 試験実施
    llm = connector.get_llm()
    response = llm.invoke("こんにちは")

    # 結果検証
    # 観点1: BaseChatModel インスタンスである
    assert isinstance(llm, BaseChatModel)
    # 観点2: invoke のレスポンスが AIMessage で content が非空
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


@pytest.mark.integration
def test_obsidian_vault_search_01(
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
) -> None:
    """obsidian_vault_search() を呼び出した時、 クエリと意味的に近いドキュメントを返せる。

    観点1: sync 後に search が例外なく完了し、ToolMessage として返る
    観点2: ToolMessage.content が非空文字列で、artifact の各 Document が path メタデータを持つ
    """
    raw_entity = vault_entities
    PgVault.sync(vault_docs, sa_engine, raw_entity)
    obsidian_retriever.sync_chunks()

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {"search_query": vault_docs[0].page_content[:20]},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    result = _make_agent(ai_msg, connector.obsidian_vault_search).invoke(
        {"messages": [HumanMessage(content="ノートを検索して")]},
        context=ContextSchema(obsidian_store=obsidian_retriever),
    )

    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_msg.artifact, list)
    assert len(tool_msg.artifact) >= 1
    # 観点2
    assert len(tool_msg.content) > 0
    for doc in tool_msg.artifact:
        assert "path" in doc.metadata


@pytest.mark.integration
def test_obsidian_vault_get_01(
    obsidian_retriever: ObsidianLlamaRetriever,
    vault_entities: type,
    sa_engine: Engine,
    vault_docs: list[Document],
) -> None:
    """obsidian_vault_get() を呼び出した時、指定した document_id のドキュメントを取得できる。存在しない document_id では空の結果を返す。

    観点1: document_id を指定するとノートを取得でき、ToolMessage として返る
    観点2: ToolMessage.content に page_content が含まれる
    観点3: 存在しない ID を指定すると ToolMessage.artifact が空リスト、content が空文字列
    """
    raw_entity = vault_entities
    PgVault.sync([vault_docs[0]], sa_engine, raw_entity)
    doc_id = vault_docs[0].metadata["document_id"]
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
    result = _make_agent(ai_msg, connector.obsidian_vault_get).invoke(
        {"messages": [HumanMessage(content="ノートを取得して")]},
        context=ContextSchema(obsidian_store=obsidian_retriever),
    )

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
    result2 = _make_agent(ai_msg2, connector.obsidian_vault_get).invoke(
        {"messages": [HumanMessage(content="nonexistent.md を取得して")]},
        context=ContextSchema(obsidian_store=obsidian_retriever),
    )
    # 観点3
    tool_msg2 = next(m for m in result2["messages"] if isinstance(m, ToolMessage))
    assert tool_msg2.artifact == []
    assert tool_msg2.content == ""
