from typing import Sequence

import mlflow
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.schema import NodeWithScore, TransformComponent
from sqlalchemy import select
from sqlalchemy.orm import Session

from assistant_agent.entities import base
from assistant_agent.utils.absclass import StoreContext


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

        # Chunking ロジック設定
        self._sa_engine = store_context.get_engine()
        self._vector_store = store_context.get_vector_store(vectorstore_name, embed_dim)
        self._docstore = store_context.get_docstore(docstore_name)

        # Pipeline 設定
        self._pipeline = IngestionPipeline(
            transformations=list(transformations) + [embed_model],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        )

    @mlflow.trace(span_type="RETRIEVER")
    def search_documents(self, query: str, top_k: int) -> Sequence[NodeWithScore]:
        """チャンク類似検索."""
        # チャンク類似検索
        index = VectorStoreIndex.from_vector_store(self._vector_store, self._embed_model)
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        # チャンクを返す
        return nodes

    def sync_chunks(self) -> None:
        """Vault テーブルの全ドキュメントを pipeline に渡して Chunk 層を更新."""
        with Session(self._sa_engine) as session:
            rows = session.scalars(select(self._vault_entity)).all()

        llama_docs = [
            Document(id_=row.document_id, text=row.content, metadata=row.document_metadata)
            for row in rows
        ]

        self._pipeline.run(documents=llama_docs)
