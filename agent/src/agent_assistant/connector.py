import json
from os.path import basename
from typing import Sequence

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from llama_index.core.vector_stores.types import MetadataFilters
from pydantic import BaseModel, Field

from agent_assistant.context import ContextSchema
from agent_assistant.retriever.obsidian_llama import ObsidianLlamaRetriever


def get_tools():
    """エージェントに使用させるツールを返す."""
    return [get_weather, obsidian_vault_search, obsidian_vault_get]


# --------------------------------
# Tool: get_weather
# --------------------------------
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# --------------------------------
# Tool: obsidian_vault_search
# --------------------------------
class ObsidianVaultSearchInput(BaseModel):
    search_query: str = Field(description="Search word")
    filters: MetadataFilters | None = Field(
        default=None,
        description=(
            "Metadata filter. Set to null if not needed.\n"
            "Available metadata fields (specified in the key of each element in filters.filters):\n"
            "- path (str): Vault-relative file path. Example: '02_Daily/2024-01-01.md'.\n"
            "  Use operator='text_match' for partial matching of folder/file names.\n"
            "- date (str): Note creation date/time. 'YYYY-MM-DD HH:MM:SS' format.\n"
            "  Use operator='>=' / '<=' / '>' / '<' for date range filtering.\n"
            "\n"
            "Valid values for operator: '==' / '>' / '<' / '!=' / '>=' / '<=' / 'text_match'\n"
            "If no filter is needed, set filters to null.\n"
        ),
    )
    full_fetch: bool = Field(
        default=False,
        description=(
            "When set to True, disables filtering by similarity score and fetches all matching filters. \n"  # noqa: E501
            "Use this when filtering only by metadata filters \n"
            "(e.g., all notes created within a specific period). \n"
            "It is forbidden to set this to True without applying a metadata filter (too much data). \n"  # noqa: E501
            "When False, performs a normal search returning only the top 10 vector similarities. \n"
        ),
    )


@tool(args_schema=ObsidianVaultSearchInput, response_format="content_and_artifact")
def obsidian_vault_search(
    search_query: str,
    filters: MetadataFilters | None,
    full_fetch: bool,
    runtime: ToolRuntime[ContextSchema],
) -> tuple[str, list[Document]]:
    """Perform vector search on Obsidian vault with metadata filters."""
    obsidian_store = runtime.context.obsidian_store
    top_k = 9999 if full_fetch else 10
    results = obsidian_store.search_documents(search_query, top_k=top_k, filters=filters)
    return format_document_ids(results), results


# --------------------------------
# Tool: obsidian_vault_get
# --------------------------------
class ObsidianVaultGetInput(BaseModel):
    document_ids: list[str] = Field(
        description="List of document_ids of the notes to retrieve (unlimited number of elements)"
    )


@tool(args_schema=ObsidianVaultGetInput, response_format="content_and_artifact")
def obsidian_vault_get(
    document_ids: Sequence[str], runtime: ToolRuntime[ContextSchema]
) -> tuple[str, Sequence[Document]]:
    """Return the text of Obsidian notes specified by document_id.

    # Note format
    ## front matter
    The response includes document_id and metadata, holding the following information:
    ```yaml
    document_id: (Document identifier)
    metadata:
        date: (Document creation date)
        path: (Document path)
        tags: (Category tags)
    ```

    ## wiki links
    Notes have zero or more links to other notes.
    Links are written in Wiki or Obsidian notation.
    Can be the linked note retrieved by calling the "obsidian_vault_get" tool with a document_id.
    The document_id can be got by matching the wikilink with forward_link in the frontmatter.
    """
    obsidian_store = runtime.context.obsidian_store
    results = obsidian_store.get_documents_by_ids(document_ids)
    return format_documents(results, obsidian_store), results


# --------------------------------
# Utilities
# --------------------------------
def format_document_ids(documents: Sequence[Document]) -> str:
    """Document リストから document_id と path の一覧文字列を返す."""
    lines = [f"Search results ({len(documents)} documents found):"]
    for doc in documents:
        lines.append(f"- {{ document_id: \"{doc.id}\", path: \"{doc.metadata['path']}\" }}")
    lines.append("\nUse obsidian_vault_get with document_ids to retrieve full content.")
    return "\n".join(lines)


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
            f"document_id: {doc.id}",
            f"metadata: {json.dumps(meta_without_links, ensure_ascii=False)}",
        ]

        if forward_links:
            linked_docs = obsidian_store.get_documents_by_ids(forward_links)
            lines.append("forward_link:")
            for linked in linked_docs:
                link_name = basename(linked.metadata.get("path", ""))
                link_id = linked.id or ""
                lines.append(f'  "{link_id}": "{link_name}"')

        lines.extend(["```\n", doc.page_content])
        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts)
