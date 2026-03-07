import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, Column
from .chunkstore.postgres import PGVectorChunkStore


class MonthlyNewsChunkStore(PGVectorChunkStore):
    def __init__(self, engine: PGEngine):
        api_key = os.getenv("ENV_GEMINI_API_KEY")
        super().__init__(
            engine,
            [
                Column("source", "text", False),
                Column("section", "text", True),
            ],
            GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key),
            3072,
        )


class ObsidianChunkStore(PGVectorChunkStore):
    """Obsidian Vault のチャンクを PGVector に格納するストア。

    各チャンクに元ノートの path (vault 相対パス) と
    原文内の開始位置 position を保持する。
    """

    def __init__(self, engine: PGEngine):
        api_key = os.getenv("ENV_GEMINI_API_KEY")
        super().__init__(
            engine,
            [
                Column("path", "text", False),
                Column("start_index", "integer", False),
            ],
            GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key),
            3072,
        )
