import os
from collections.abc import Sequence
from typing import Any

import mlflow
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from assistant_agent.entities import base
from assistant_agent.entities import postgres as entities
from assistant_agent.utils.absclass import StoreConnector
from assistant_agent.utils.context import ContextRegistry


class VaultSampleRetriever:
    """レトリーバー参考実装."""

    def __init__(
        self,
        chunk_entity: type,
        store_conn: StoreConnector,
        splitter: TextSplitter,
        embed_model: Embeddings,
        vault_entity: type[base.DocumentFields],
    ):
        """Construct VaultSampleRetriever.

        Args:
            chunk_entity: チャンクテーブルに対応する ORM エンティティクラス。
            store_conn: ベクターストアを生成するファクトリ。
            splitter: チャンク分割器。
            embed_model: テキスト埋め込みモデル。
            vault_entity: Vault テーブルに対応する ORM エンティティクラス。

        """
        self._vault_entity = vault_entity
        self._embed_model = embed_model
        self._store_conn = store_conn
        self._chunk_entity = chunk_entity
        self._splitter = splitter
        self._initialized: bool = False

    _sa_engine: AsyncEngine

    async def initialize(self) -> None:
        """ストアを遅延初期化する。2回目以降の呼び出しはスキップ.

        派生クラスでオーバーライドする場合は super().initialize() を呼び出すこと。
        """
        if self._initialized:
            return
        self._sa_engine = self._store_conn.get_engine()
        self._vector_store = await PGVectorStore.create(
            engine=PGEngine.from_engine(self._sa_engine),
            embedding_service=self._embed_model,
            table_name=self._chunk_entity.__tablename__,
            schema_name=self._chunk_entity.metadata.schema,
            # 空リストは falsy 判定されるため、metadata_columns 自動認識を確実に発動させる目的で
            # metadata_json_column（JSON 列としての扱いは変わらない）を明示的に指定する
            ignore_metadata_columns=["langchain_metadata"],
        )
        self._initialized = True

    @mlflow.trace(span_type="RETRIEVER")
    async def search_documents(self, query: str, top_k: int) -> Sequence[Document]:
        """チャンク類似検索."""
        await self.initialize()
        return await self._vector_store.asimilarity_search(query, k=top_k)

    async def sync_chunks(self) -> None:
        """Vault テーブルと Chunk テーブルの差分のみを分割し、Chunk 層を更新."""
        await self.initialize()
        diff_rows = await base.VaultUtils.sync_chunks(
            self._vault_entity, self._chunk_entity, self._sa_engine
        )

        docs = [
            Document(
                id=row.document_id,
                page_content=row.content,
                metadata={
                    **row.document_metadata,
                    "document_id": row.document_id,
                    "document_content_hash": row.document_content_hash,
                },
            )
            for row in diff_rows
        ]
        chunks = self._splitter.split_documents(docs)
        await self._vector_store.aadd_documents(chunks)


@ContextRegistry.register("sample_retriever")
def build(store_conn: StoreConnector | None = None, **_: Any) -> VaultSampleRetriever | None:
    """Sample レトリーバーを生成する."""
    if store_conn is None:
        return None
    api_key = os.getenv("AA_GEMINI_API_KEY")
    assert api_key is not None
    retriever = VaultSampleRetriever(
        chunk_entity=entities.SampleChunkEntity,
        store_conn=store_conn,
        splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80),
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(api_key),
        ),
        vault_entity=entities.SampleEntity,
    )
    return retriever
