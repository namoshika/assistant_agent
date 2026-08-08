import os
from collections.abc import Sequence
from typing import TypedDict

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

import assistant_agent.utils.format as fmt
from assistant_agent.services import VaultSampleRetriever


class SampleContext(TypedDict):
    sample_retriever: VaultSampleRetriever


# --------------------------------
# Tool: sample_search
# --------------------------------
class SampleSearchInput(BaseModel):
    search_query: str = Field(
        description="Search word (At least 1 character required).", default=" ", min_length=1
    )


@tool(args_schema=SampleSearchInput, response_format="content_and_artifact")
async def sample_search(
    search_query: str,
    runtime: ToolRuntime[SampleContext],
) -> tuple[str, Sequence[Document]]:
    """Perform vector search on web pages saved in Local DB."""
    retriever = runtime.context["sample_retriever"]
    results = await retriever.search_documents(search_query, top_k=10)
    return fmt.format_doc_list(
        [
            fmt.ContentsWithFrontmatter(
                title=os.path.basename(item.metadata["file_path"]),
                contents=item.page_content,
                frontmatter=None,
            )
            for item in results
        ]
    ), results
