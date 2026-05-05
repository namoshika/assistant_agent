import os
from typing import Any, Sequence

import mlflow
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, MetadataFilters
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from assistant_agent.entities import base
from assistant_agent.entities import postgres as entities
from assistant_agent.utils.absclass import StoreConnector
from assistant_agent.utils.context import ContextRegistry


class VaultObsidianRetriever:
    """Obsidian Vault レトリーバー."""

    def __init__(
        self,
        docstore_name: str,
        vectorstore_name: str,
        store_conn: StoreConnector,
        transformations: Sequence[TransformComponent],
        embed_model: BaseEmbedding,
        embed_dim: int,
        vault_entity: type[base.ObsidianFields],
    ):
        """Construct VaultObsidianRetriever.

        Args:
            docstore_name: ドキュメントストアの識別名。
            vectorstore_name: ベクターストアの識別名。
            store_conn: ベクターストア、ドキュメントストアを生成するファクトリ。
            transformations: ドキュメントの変換処理リスト (埋め込みを除く)。
            embed_model: テキスト埋め込みモデル。
            embed_dim: 埋め込みベクトルの次元数。
            vault_entity: Vault テーブルに対応する ORM エンティティクラス。

        """
        self._vault_entity = vault_entity
        self._embed_model = embed_model
        self._store_conn = store_conn
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
        self._sa_engine = self._store_conn.get_engine()
        self._vector_store = self._store_conn.get_vector_store(
            self._vectorstore_name, self._embed_dim
        )
        self._docstore = self._store_conn.get_docstore(self._docstore_name)
        self._pipeline = IngestionPipeline(
            transformations=list(self._transformations) + [self._embed_model],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        )
        self._initialized = True

    @mlflow.trace(span_type="RETRIEVER")
    def search_documents(self, query: str, top_k: int, **kwargs: Any) -> Sequence[Document]:
        """チャンク類似検索 → document_id 重複除去して返す.

        返却する Document は id_ と metadata のみを保持し、text は空文字列である。
        全文が必要な場合は get_documents_by_ids を使用すること。
        """
        self.initialize()
        filters: MetadataFilters | None = kwargs.get("filters")

        # チャンク類似検索
        index = VectorStoreIndex.from_vector_store(self._vector_store, self._embed_model)
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        nodes = retriever.retrieve(query)

        # document_id 重複除去しつつ Document (id_ + metadata のみ) を構築
        seen: dict[str, Document] = {}
        for n in nodes:
            doc_id = n.node.ref_doc_id
            if doc_id is not None and doc_id not in seen:
                seen[doc_id] = Document(id_=doc_id, text="", metadata=n.node.metadata)
        return list(seen.values())

    @mlflow.trace(span_type="RETRIEVER")
    def get_documents_by_ids(self, document_ids: Sequence[str]) -> Sequence[Document]:
        """document_id の完全一致する Document を取得する."""
        self.initialize()
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
        self.initialize()
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
        self.initialize()
        with Session(self._sa_engine) as session:
            rows = session.scalars(select(self._vault_entity)).all()

        llama_docs = [
            Document(
                id_=row.document_id,
                text=row.content,
                metadata={
                    k: v if isinstance(v, (str, int, float)) or v is None else str(v)
                    for k, v in row.document_metadata.items()
                    if k != "forward_links"
                },
            )
            for row in rows
        ]

        self._pipeline.run(documents=llama_docs)


@ContextRegistry.register("obsidian_retriever")
def build(store_conn: StoreConnector | None = None, **_: Any) -> VaultObsidianRetriever | None:
    """Obsidian レトリーバーを生成する (chunk_size=1024)."""
    if store_conn is None:
        return None
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    assert env_gemini_api_key is not None

    return VaultObsidianRetriever(
        docstore_name="obsidian_docs",
        vectorstore_name="obsidian_vectors",
        store_conn=store_conn,
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200, paragraph_separator="\n\n")
        ],
        embed_model=GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=env_gemini_api_key,
        ),
        embed_dim=3072,
        vault_entity=entities.ObsidianEntity,
    )


@ContextRegistry.register("obsidian_retriever", variant="chunk_512")
def build_c512(store_conn: StoreConnector | None = None, **_: Any) -> VaultObsidianRetriever | None:
    """Obsidian レトリーバーを生成する (chunk_size=512)."""
    if store_conn is None:
        return None
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    assert env_gemini_api_key is not None

    return VaultObsidianRetriever(
        docstore_name="obsidian_docs",
        vectorstore_name="obsidian_vectors",
        store_conn=store_conn,
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=80, paragraph_separator="\n\n")
        ],
        embed_model=GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=env_gemini_api_key,
        ),
        embed_dim=3072,
        vault_entity=entities.ObsidianEntity,
    )
