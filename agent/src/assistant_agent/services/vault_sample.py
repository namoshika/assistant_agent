import os
from typing import Any, Sequence

import mlflow
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import SecretStr
from sqlalchemy import Engine

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

    _sa_engine: Engine

    def initialize(self) -> None:
        """ストアを遅延初期化する。2回目以降の呼び出しはスキップ.

        派生クラスでオーバーライドする場合は super().initialize() を呼び出すこと。
        """
        if self._initialized:
            return
        self._sa_engine = self._store_conn.get_engine()
        self._vector_store = self._store_conn.get_vector_store(
            self._chunk_entity, self._embed_model
        )
        self._initialized = True

    @mlflow.trace(span_type="RETRIEVER")
    def search_documents(self, query: str, top_k: int) -> Sequence[Document]:
        """チャンク類似検索."""
        self.initialize()
        return self._vector_store.similarity_search(query, k=top_k)

    def sync_chunks(self) -> None:
        """Vault テーブルと Chunk テーブルの差分のみを分割し、Chunk 層を更新."""
        self.initialize()
        diff_rows = base.VaultUtils.sync_chunks(
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
        self._vector_store.add_documents(chunks)


@ContextRegistry.register("sample_retriever")
def build(store_conn: StoreConnector | None = None, **_: Any) -> VaultSampleRetriever | None:
    """Sample レトリーバーを生成する."""
    if store_conn is None:
        return None
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    assert env_gemini_api_key is not None
    retriever = VaultSampleRetriever(
        chunk_entity=entities.SampleChunkEntity,
        store_conn=store_conn,
        splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80),
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(env_gemini_api_key),
        ),
        vault_entity=entities.SampleEntity,
    )
    return retriever
