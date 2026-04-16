from typing import Any, Sequence

import mlflow
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import MetadataFilters
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from assistant_agent.entities import base
from assistant_agent.utils.absclass import StoreContext

# 日本語テキスト向け区切り文字（TextChunker._JAPANESE_SEPARATORS と同等）
_JAPANESE_PARAGRAPH_SEP = "\n\n"


class VaultObsidianRetriever:
    """LlamaIndex IngestionPipeline を使った Obsidian Vault レトリーバー.

    docstore_name / vectorstore_name / chunk_size / chunk_overlap を
    コンストラクタで指定することで、複数の設定のインスタンスを作成できる。
    """

    def __init__(
        self,
        sa_engine: Engine,
        store_factory: StoreContext,
        docstore_name: str,
        vectorstore_name: str,
        embed_model: BaseEmbedding,
        embed_dim: int,
        vault_entity: type[base.ObsidianFields],
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ):
        """Construct VaultObsidianRetriever."""
        self._sa_engine = sa_engine
        self._vault_entity = vault_entity
        self._embed_model = embed_model

        # Chunking ロジック設定
        self._vector_store = store_factory.get_vector_store(vectorstore_name, embed_dim)
        self._docstore = store_factory.get_docstore(docstore_name)

        # Pipeline 設定
        self._pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    paragraph_separator=_JAPANESE_PARAGRAPH_SEP,
                ),
                self._embed_model,
            ],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        )

    @mlflow.trace(span_type="RETRIEVER")
    def search_documents(self, query: str, top_k: int, **kwargs: Any) -> Sequence[Document]:
        """チャンク類似検索 → document_id 重複除去 → raw から全文取得."""
        filters: MetadataFilters | None = kwargs.get("filters")

        # チャンク類似検索
        index = VectorStoreIndex.from_vector_store(self._vector_store, self._embed_model)
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        nodes = retriever.retrieve(query)

        # document_id 重複除去
        sorted_ids = list(
            dict.fromkeys(n.node.ref_doc_id for n in nodes if n.node.ref_doc_id is not None)
        )

        # raw から全文取得
        return self.get_documents_by_ids(sorted_ids)

    @mlflow.trace(span_type="RETRIEVER")
    def get_documents_by_ids(self, document_ids: Sequence[str]) -> Sequence[Document]:
        """document_id の完全一致する Document を取得する."""
        with Session(self._sa_engine) as session:
            rows = session.scalars(
                select(self._vault_entity).where(self._vault_entity.document_id.in_(document_ids))
            ).all()

        # raw から全文取得
        id_to_doc = {
            row.document_id: Document(
                id_=row.document_id,
                text=row.content,
                metadata=row.document_metadata,
            )
            for row in rows
        }
        return [id_to_doc[doc_id] for doc_id in document_ids if doc_id in id_to_doc]

    def get_backlinks(self, document_id: str) -> Sequence[Document]:
        """document_id のノートにリンクしているノートを返す（バックリンク）.

        doc_metadata["forward_links"] は document_id のリストを格納している前提。
        document_id が空文字列の場合は ValueError を raise する。
        """
        if not document_id:
            raise ValueError("document_id must not be empty")
        with Session(self._sa_engine) as session:
            rows = session.scalars(
                select(self._vault_entity).where(self._vault_entity.backlink_filter(document_id))
            ).all()
        return [
            Document(
                id_=row.document_id,
                text=row.content,
                metadata=row.document_metadata,
            )
            for row in rows
        ]

    def sync_chunks(self) -> None:
        """Vault テーブルの全ドキュメントを pipeline に渡して Chunk 層を更新."""
        with Session(self._sa_engine) as session:
            rows = session.scalars(select(self._vault_entity)).all()

        llama_docs = [
            Document(
                id_=row.document_id,
                text=row.content,
                metadata={k: v for k, v in row.document_metadata.items() if k != "forward_links"},
            )
            for row in rows
        ]

        self._pipeline.run(documents=llama_docs)
