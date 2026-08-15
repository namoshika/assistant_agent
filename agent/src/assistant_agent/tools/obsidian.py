import logging
import os
from collections.abc import Sequence
from typing import TypedDict

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from assistant_agent.services import VaultObsidianRetriever
from assistant_agent.services.vault_obsidian import SearchFilters
from assistant_agent.utils.format import ContentsWithFrontmatter, format_doc_ids, format_doc_list


class ObsidianContext(TypedDict):
    obsidian_retriever: VaultObsidianRetriever


# --------------------------------
# Tool: obsidian_vault_search
# --------------------------------
class SearchToolInput(BaseModel):
    search_query: str = Field(
        description=(
            "Search word (At least 1 character required). "
            "This field cannot be empty even when full_fetch=True or when filtering only by "
            "metadata filters; use the default single-space value in that case."
        ),
        default=" ",
        min_length=1,
    )
    filters: SearchFilters | None = Field(
        default=None,
        description=(
            "Metadata filter. Set to null if not needed. "
            "Never invent a file_path or date value that was not explicitly given by the user; "
            "leave this null instead."
        ),
    )
    full_fetch: bool = Field(
        default=False,
        description=(
            "Keep this False for normal searches (default: top 10 vector similarities). \n"
            "Only set to True when the user explicitly asks for ALL notes matching a metadata "
            "filter (e.g., all notes created within a specific period) and you have set a "
            "concrete filters value from information the user actually provided. \n"
            "It is forbidden to set this to True without such a user-provided filter "
            "(too much data)."
        ),
    )


@tool(args_schema=SearchToolInput, response_format="content_and_artifact")
async def obsidian_vault_search(
    search_query: str,
    filters: SearchFilters | None,
    full_fetch: bool,
    runtime: ToolRuntime[ObsidianContext],
) -> tuple[str, Sequence[Document] | Exception]:
    """Perform vector search on Obsidian vault with metadata filters."""
    retriever = runtime.context["obsidian_retriever"]
    top_k = 9999 if full_fetch else 10
    try:
        results = await retriever.search_documents(search_query, top_k=top_k, filters=filters)
        return format_doc_ids(results), results
    except Exception as ex:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to search Obsidian vault")
        return str(ex), ex


# --------------------------------
# Tool: obsidian_vault_get
# --------------------------------
class GetToolInput(BaseModel):
    document_ids: list[str] = Field(
        description="List of document_ids of the notes to retrieve (unlimited number of elements)"
    )


@tool(args_schema=GetToolInput, response_format="content_and_artifact")
async def obsidian_vault_get(
    document_ids: Sequence[str], runtime: ToolRuntime[ObsidianContext]
) -> tuple[str, Sequence[Document] | Exception]:
    """Return the text of Obsidian notes specified by document_id.

    # Note format
    ## front matter
    The response includes document_id and metadata, holding the following information:
    ```yaml
    document_id: (Document identifier)
    metadata:
        date: (Document creation date)
        file_path: (Document path)
        tags: (Category tags)
    ```

    ## wiki links
    Notes have zero or more links to other notes.
    Links are written in Wiki or Obsidian notation.
    Can be the linked note retrieved by calling the "obsidian_vault_get" tool with a document_id.
    The document_id can be got by matching the wikilink with forward_link in the frontmatter.
    """
    retriever = runtime.context["obsidian_retriever"]
    try:
        docs = await retriever.get_documents_by_ids(document_ids)

        contents = [
            ContentsWithFrontmatter(
                title=os.path.basename(doc.metadata["file_path"]),
                contents=doc.page_content,
                frontmatter={
                    k: await _format_links(v, retriever) if k == "forward_links" else v
                    for k, v in doc.metadata.items()
                },
            )
            for doc in docs
        ]
        return format_doc_list(contents), docs
    except Exception as ex:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to get Obsidian vault notes")
        return str(ex), ex


async def _format_links(
    document_ids: list[str], retriever: VaultObsidianRetriever
) -> dict[str, str]:
    result: dict[str, str] = {}
    for doc in await retriever.get_documents_by_ids(document_ids):
        assert doc.id is not None
        result[doc.id] = os.path.basename(doc.metadata["file_path"])
    return result
