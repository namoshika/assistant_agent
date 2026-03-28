from typing import Sequence

from langchain_core.documents import Document
from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import SecretStr
from sqlalchemy import Engine, cast, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session

from agent_assistant.model import ObsidianVaultRawEntity
from agent_assistant.utils.absclass import DocumentRetriever

# 日本語テキスト向け区切り文字（TextChunker._JAPANESE_SEPARATORS と同等）
_JAPANESE_PARAGRAPH_SEP = "\n\n"
_JAPANESE_CHUNKING_REGEX = r"[。？、！．，　\u200b\n ]"


class ObsidianLlamaRetriever(DocumentRetriever):
    """LlamaIndex IngestionPipeline を使った Obsidian Vault レトリーバー.

    docstore_name / vectorstore_name / chunk_size / chunk_overlap を
    コンストラクタで指定することで、複数の設定のインスタンスを作成できる。
    """

    def __init__(
        self,
        sa_engine: Engine,
        connection_string: str,
        emb_api_key: SecretStr,
        docstore_name: str,
        vectorstore_name: str,
        schema_name: str = "public",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        vault_entity: type[ObsidianVaultRawEntity] = ObsidianVaultRawEntity,
    ):
        """Construct ObsidianLlamaRetriever."""
        self._sa_engine = sa_engine
        self._vault_entity = vault_entity
        self._embed_model = GoogleGenAIEmbedding(
            model="gemini-embedding-001",
            api_key=emb_api_key.get_secret_value(),
        )

        # Chunking ロジック設定
        url = make_url(connection_string)
        self._vector_store = PGVectorStore.from_params(
            host=url.host,
            port=str(url.port or 5432),
            database=url.database,
            user=url.username,
            password=str(url.password or ""),
            table_name=vectorstore_name,
            embed_dim=3072,
            schema_name=schema_name,
            use_jsonb=True,
        )
        self._docstore = PostgresDocumentStore.from_params(
            host=url.host,
            port=str(url.port or 5432),
            database=url.database,
            user=url.username,
            password=str(url.password or ""),
            table_name=docstore_name,
            schema_name=schema_name,
            use_jsonb=True,
        )

        # Pipeline 設定
        self._pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    paragraph_separator=_JAPANESE_PARAGRAPH_SEP,
                    secondary_chunking_regex=_JAPANESE_CHUNKING_REGEX,
                ),
                self._embed_model,
            ],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        )
        self._index: VectorStoreIndex | None = None

    def search_documents(self, query: str, top_k: int) -> list[Document]:
        """チャンク類似検索 → document_id 重複除去 → raw から全文取得."""
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(self._vector_store, self._embed_model)
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        sorted_ids = list(dict.fromkeys(n.node.ref_doc_id for n in nodes))

        with Session(self._sa_engine) as session:
            rows = session.scalars(
                select(self._vault_entity).where(self._vault_entity.document_id.in_(sorted_ids))
            ).all()

        id_to_doc = {
            row.document_id: Document(
                page_content=row.content,
                id=row.document_id,
                metadata=row.document_metadata,
            )
            for row in rows
        }
        return [id_to_doc[doc_id] for doc_id in sorted_ids if doc_id in id_to_doc]

    def get_documents_by_ids(self, document_ids: Sequence[str]) -> Sequence[Document]:
        """document_id の完全一致する Document を取得する."""
        with Session(self._sa_engine) as session:
            rows = session.scalars(
                select(self._vault_entity).where(self._vault_entity.document_id.in_(document_ids))
            ).all()
        return [
            Document(
                id=row.document_id,
                page_content=row.content,
                metadata=row.document_metadata,
            )
            for row in rows
        ]

    def get_backlinks(self, document_id: str) -> Sequence[Document]:
        """document_id のノートにリンクしているノートを返す（バックリンク）.

        doc_metadata["forward_links"] は document_id のリストを格納している前提。
        document_id が空文字列の場合は ValueError を raise する。
        """
        if not document_id:
            raise ValueError("document_id must not be empty")
        with Session(self._sa_engine) as session:
            rows = session.scalars(
                select(self._vault_entity).where(
                    cast(self._vault_entity.document_metadata["forward_links"], JSONB).contains(
                        [document_id]
                    )
                )
            ).all()
        return [
            Document(
                id=row.document_id,
                page_content=row.content,
                metadata=row.document_metadata,
            )
            for row in rows
        ]

    def sync_chunks(self) -> None:
        """Vault テーブルの全ドキュメントを pipeline に渡して Chunk 層を更新."""
        with Session(self._sa_engine) as session:
            rows = session.scalars(select(self._vault_entity)).all()

        llama_docs = [
            LlamaDocument(
                doc_id=row.document_id,
                text=row.content,
                metadata=row.document_metadata,
            )
            for row in rows
        ]

        self._pipeline.run(documents=llama_docs)
        self._index = None  # キャッシュを無効化
