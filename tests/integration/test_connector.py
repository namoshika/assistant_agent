import os
import pytest
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent

from agent_assistant import connector
from agent_assistant.context import ContextSchema
from agent_assistant.retriever.obsidian import (
    ObsidianDocumentStore,
    path_to_document_id,
)
from tests.integration.conftest import make_docs


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
def test_get_llm_01() -> None:
    """get_llm() が返す BaseChatModel を invoke するとレスポンスが返る。

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
def test_obsidian_vault_search_01(obsidian_store: ObsidianDocumentStore) -> None:
    """import 後に search_documents() で意味的に近いドキュメントが返る。

    観点1: import 後に search が例外なく完了し、ToolMessage として返る
    観点2: ToolMessage.content が非空文字列で、artifact の各 Document が path メタデータを持つ
    """
    obsidian_store.connect()
    obsidian_store.import_documents(make_docs())

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "obsidian_vault_search",
                "args": {"search_query": "LLM フレームワーク"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    result = _make_agent(ai_msg, connector.obsidian_vault_search).invoke(
        {"messages": [HumanMessage(content="LLM フレームワークを検索して")]},
        context=ContextSchema(obsidian_store=obsidian_store),
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
def test_obsidian_vault_get_01(obsidian_store: ObsidianDocumentStore) -> None:
    """ファイル名のみ指定でノートを取得し、存在しない ID では空の結果を返す。

    観点1: ファイル名のみ指定でノートを取得でき、ToolMessage として返る
    観点2: ToolMessage.content に page_content が含まれる
    観点3: 存在しない ID を指定すると ToolMessage.artifact が空リスト、content が空文字列
    """
    obsidian_store.connect()
    obsidian_store.import_documents(make_docs())
    doc_id = path_to_document_id("langchain.md")
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
        {"messages": [HumanMessage(content="langchain.md を取得して")]},
        context=ContextSchema(obsidian_store=obsidian_store),
    )

    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert len(tool_msg.artifact) == 1
    # 観点2
    assert "LangChain" in tool_msg.content
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
        context=ContextSchema(obsidian_store=obsidian_store),
    )
    # 観点3
    tool_msg2 = next(m for m in result2["messages"] if isinstance(m, ToolMessage))
    assert tool_msg2.artifact == []
    assert tool_msg2.content == ""
