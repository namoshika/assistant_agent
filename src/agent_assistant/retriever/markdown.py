from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, Column
from pydantic import SecretStr

from agent_assistant.utils import absclass
from agent_assistant.utils.chunkstore.postgres import PGVectorChunkStore


class MarkdownChunkStore(PGVectorChunkStore):
    def __init__(self, engine: PGEngine, emb_api_key: SecretStr):
        super().__init__(
            engine,
            [
                Column("source", "text", False),
                Column("section", "text", True),
            ],
            GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", api_key=emb_api_key
            ),
            3072,
        )


class MarkdownDocumentStore(absclass.DocumentStore):
    def __init__(
        self,
        store_name: str,
        reader: absclass.ChunkReader,
        writer: absclass.ChunkWriter,
        chunker: absclass.DocumentChunker,
    ):
        self._store_name = store_name
        self._reader = reader
        self._writer = writer
        self._chunker = chunker
        self._store = None

    def connect(self) -> None:
        if self._store is not None:
            return
        self._store = self._reader.get_vectorstore(self._store_name)

    def search_documents(self, query: str, top_k: int) -> list[Document]:
        if self._reader is None:
            raise ValueError("Reader is not set")
        if self._store is None:
            raise ValueError("Store is not connected")

        return self._store.similarity_search(query, k=top_k)

    def import_documents(self, documents: list[Document]) -> None:
        if self._writer is None:
            raise ValueError("Writer is not set")

        chunked = self._chunker.chunk(documents)
        self._writer.add_chunks(self._store_name, chunked)
