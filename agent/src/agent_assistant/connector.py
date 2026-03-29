import json
import os
from os.path import basename
from typing import Annotated, Sequence

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from agent_assistant.context import ContextSchema
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


def get_llm() -> BaseChatModel:
    """エージェントが使用するLLMを返す."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")  # noqa: F841
    assert aws_access_key_id is not None
    assert aws_secret_access_key is not None
    aws_access_key_id = SecretStr(aws_access_key_id)
    aws_secret_access_key = SecretStr(aws_secret_access_key)

    # LLM 作成
    # from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    # llm = ChatGoogleGenerativeAI(
    #     model=os.environ.get("ENV_GEMINI_MODEL_ID", "gemini-3-flash-preview"),
    #     api_key=os.environ.get("ENV_GEMINI_API_KEY"),
    #     thinking_level="minimal",
    # )

    # LLM 作成
    from langchain_aws import ChatBedrockConverse

    llm = ChatBedrockConverse(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_default_region,
    )
    return llm


def get_tools():
    """エージェントに使用させるツールを返す."""
    return [get_weather, obsidian_vault_search, obsidian_vault_get]


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool(response_format="content_and_artifact")
def obsidian_vault_search(
    search_query: Annotated[str, "検索クエリ"],
    runtime: ToolRuntime[ContextSchema],
) -> tuple[str, list[Document]]:
    """Obsidian vault を検索し、マッチしたノートの原文を返す."""
    obsidian_store = runtime.context.obsidian_store
    results = obsidian_store.search_documents(search_query, top_k=5)
    return format_documents(results, obsidian_store), results


@tool(response_format="content_and_artifact")
def obsidian_vault_get(
    document_ids: Annotated[list[str], "取得するノートの document_id のリスト"],
    runtime: ToolRuntime[ContextSchema],
) -> tuple[str, Sequence[Document]]:
    """document_id で指定した Obsidian ノートの原文を返す."""
    obsidian_store = runtime.context.obsidian_store
    results = obsidian_store.get_documents_by_ids(document_ids)
    return format_documents(results, obsidian_store), results


def format_documents(documents: Sequence[Document], obsidian_store: ObsidianLlamaRetriever) -> str:
    """Document オブジェクトのリストを、エージェントが読みやすいテキスト形式に整形する.

    Args:
        documents: 整形対象の Document リスト。
        obsidian_store: リンク情報の解決に使用するレトリーバー。

    Returns:
        メタデータ、前方リンク、および本文を含む整形済み文字列。

    """
    parts = []
    for doc in documents:
        meta = doc.metadata
        forward_links: list[str] = meta.get("forward_links") or []
        meta_without_links = {k: v for k, v in meta.items() if k != "forward_links"}

        lines = [
            f"title: {basename(meta['path'])}",
            "===\n",
            "```yaml",
            f"document_id: {meta['document_id']}",
            f"metadata: {json.dumps(meta_without_links, ensure_ascii=False)}",
        ]

        if forward_links:
            linked_docs = obsidian_store.get_documents_by_ids(forward_links)
            lines.append("forward_link:")
            for linked in linked_docs:
                link_name = basename(linked.metadata.get("path", ""))
                link_id = linked.metadata.get("document_id", "")
                lines.append(f'  "{link_id}": "{link_name}"')

        lines.extend(["```\n", doc.page_content])
        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts)
