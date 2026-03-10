import os
from typing import Annotated
from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain.tools import tool, ToolRuntime
from pydantic import SecretStr
from agent_assistant.context import ContextSchema

def get_llm() -> BaseChatModel:
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    assert AWS_ACCESS_KEY_ID is not None
    assert AWS_SECRET_ACCESS_KEY is not None
    AWS_ACCESS_KEY_ID = SecretStr(AWS_ACCESS_KEY_ID)
    AWS_SECRET_ACCESS_KEY = SecretStr(AWS_SECRET_ACCESS_KEY)

    # LLM 作成
    # from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(
    #     model=os.environ.get("ENV_GEMINI_MODEL_ID", "gemini-3.1-pro-preview"),
    #     api_key=os.environ.get("ENV_GEMINI_API_KEY"),
    # )

    llm = ChatBedrock(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region=AWS_DEFAULT_REGION,
    )
    return llm


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

def get_tools():
    return [get_weather, obsidian_vault_search, obsidian_vault_get]
