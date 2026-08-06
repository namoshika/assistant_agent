from deepagents.middleware.subagents import SubAgent
from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyGetResearch,
    TavilyMap,
    TavilyResearch,
    TavilySearch,
)

_WEB_RESEARCHER_SYSTEM_PROMPT = """\
あなたは Web 調査を専門とするサブエージェントです。委譲されたタスクに対し、
Tavily 系ツール（検索・クロール・抽出・調査）を使って Web 上の情報を調査してください。

調査結果が大きい場合は本文をそのまま返さず、`write_file` でファイルへ保存し、
保存先のパスと内容の要約のみを呼び出し元へ返してください。
"""

web_researcher: SubAgent = {
    "name": "web-researcher",
    "description": "Web 検索・クロール・調査を行い、要約結果を返すサブエージェント。",
    "system_prompt": _WEB_RESEARCHER_SYSTEM_PROMPT,
    "tools": [
        TavilySearch(topic="general", country="japan"),
        TavilyExtract(),
        TavilyCrawl(),
        TavilyMap(),
        TavilyResearch(),
        TavilyGetResearch(),
    ],
}
