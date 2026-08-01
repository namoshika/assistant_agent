import os
from collections.abc import Sequence
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

import assistant_agent.utils.format as fmt
from assistant_agent.services import VaultSampleRetriever


class SampleContext(TypedDict):
    sample_retriever: VaultSampleRetriever


# --------------------------------
# Tool: get_weather
# --------------------------------
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# --------------------------------
# Tool: get_datetime_now
# --------------------------------
@tool
def get_datetime_now() -> str:
    """現在時刻を JST の ISO8601 文字列で取得する."""
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


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
