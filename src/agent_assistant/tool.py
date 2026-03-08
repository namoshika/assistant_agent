from typing import Annotated
from langchain_core.documents import Document
from langchain.tools import tool, ToolRuntime
from agent_assistant.context import ContextSchema


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool(response_format="content_and_artifact")
def obsidian_vault_search(
    search_query: Annotated[str, "検索クエリ"],
    runtime: ToolRuntime[ContextSchema],
) -> tuple[str, list[Document]]:
    """Obsidian vault を検索し、マッチしたノートの原文を返す。"""
    obsidian_store = runtime.context.obsidian_store
    obsidian_store.connect()

    tmpl = " title: {doc_name}  \n" + "===  \n" + "{doc_content}  \n\n"
    results = obsidian_store.search_documents(search_query, top_k=5)
    contents = [
        tmpl.format(
            doc_name=item.metadata.get("path", ""),
            doc_content=item.page_content,
        )
        for item in results
    ]
    return "".join(contents), results


@tool(response_format="content_and_artifact")
def obsidian_vault_get(
    ids: Annotated[list[str], "取得するノート ID のリスト"],
    runtime: ToolRuntime[ContextSchema],
) -> tuple[str, list[Document]]:
    """ファイル名または相対パスで指定した Obsidian ノートの原文を返す。"""
    obsidian_store = runtime.context.obsidian_store
    obsidian_store.connect()

    tmpl = " title: {doc_name}  \n" + "===  \n" + "{doc_content}  \n\n"
    results = obsidian_store.get_documents(ids)
    contents = [
        tmpl.format(
            doc_name=item.metadata.get("path", ""),
            doc_content=item.page_content,
        )
        for item in results
    ]
    return "".join(contents), results
