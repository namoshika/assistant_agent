import os
from typing import Any, Sequence

import mlflow
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TransformComponent
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from assistant_agent.entities import base
from assistant_agent.entities import postgres as entities
from assistant_agent.utils.absclass import StoreContext
from assistant_agent.utils.context import ContextRegistry


class VaultSampleRetriever:
    """レトリーバー参考実装."""

    def __init__(
        self,
        docstore_name: str,
        vectorstore_name: str,
        store_context: StoreContext,
        transformations: Sequence[TransformComponent],
        embed_model: BaseEmbedding,
        embed_dim: int,
        vault_entity: type[base.DocumentFields],
    ):
        """Construct VaultSampleRetriever.

        Args:
            docstore_name: ドキュメントストアの識別名。
            vectorstore_name: ベクターストアの識別名。
            store_context: ベクターストア、ドキュメントストアを生成するファクトリ。
            transformations: ドキュメントの変換処理リスト (埋め込みを除く)。
            embed_model: テキスト埋め込みモデル。
            embed_dim: 埋め込みベクトルの次元数。
            vault_entity: Vault テーブルに対応する ORM エンティティクラス。

        """
        self._vault_entity = vault_entity
        self._embed_model = embed_model
        self._store_context = store_context
        self._docstore_name = docstore_name
        self._vectorstore_name = vectorstore_name
        self._transformations = transformations
        self._embed_dim = embed_dim
        self._initialized: bool = False

    _sa_engine: Engine
    _vector_store: BasePydanticVectorStore
    _docstore: BaseDocumentStore
    _pipeline: IngestionPipeline

    def initialize(self) -> None:
        """ストアと Pipeline を遅延初期化する。2回目以降の呼び出しはスキップ.

        派生クラスでオーバーライドする場合は super().initialize() を呼び出すこと。
        """
        if self._initialized:
            return
        self._sa_engine = self._store_context.get_engine()
        self._vector_store = self._store_context.get_vector_store(
            self._vectorstore_name, self._embed_dim
        )
        self._docstore = self._store_context.get_docstore(self._docstore_name)
        self._pipeline = IngestionPipeline(
            transformations=list(self._transformations) + [self._embed_model],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        )
        self._initialized = True

    @mlflow.trace(span_type="RETRIEVER")
    def search_documents(self, query: str, top_k: int) -> Sequence[NodeWithScore]:
        """チャンク類似検索."""
        self.initialize()
        index = VectorStoreIndex.from_vector_store(self._vector_store, self._embed_model)
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        return nodes

    def sync_chunks(self) -> None:
        """Vault テーブルの全ドキュメントを pipeline に渡して Chunk 層を更新."""
        self.initialize()
        with Session(self._sa_engine) as session:
            rows = session.scalars(select(self._vault_entity)).all()

        llama_docs = [
            Document(id_=row.document_id, text=row.content, metadata=row.document_metadata)
            for row in rows
        ]

        self._pipeline.run(documents=llama_docs)


@ContextRegistry.register("sample_retriever")
def build(store_ctx: StoreContext, **_: Any) -> VaultSampleRetriever:
    """Sample レトリーバーを生成する."""
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    assert env_gemini_api_key is not None
    retriever = VaultSampleRetriever(
        "sample_docstore",
        "sample_vectors",
        store_context=store_ctx,
        transformations=[
            SentenceSplitter(chunk_size=800, chunk_overlap=80, paragraph_separator="\n\n")
        ],
        embed_model=GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=env_gemini_api_key,
        ),
        embed_dim=3072,
        vault_entity=entities.SampleEntity,
    )
    return retriever
