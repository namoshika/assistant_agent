import os
from collections.abc import Sequence
from typing import Any

import mlflow
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from assistant_agent.entities import base
from assistant_agent.entities import postgres as entities
from assistant_agent.utils.absclass import StoreConnector
from assistant_agent.utils.context import ContextRegistry


class FileFilter(BaseModel):
    """file_path フィールドに対するフィルタ条件."""

    model_config = ConfigDict(validate_by_name=True)
    like: str | None = Field(
        None,
        alias="$like",
        description=(
            "SQL LIKE pattern. Add '%' wildcards around the value for partial matching "
            "(e.g. '%02_Daily%'). Without wildcards, it matches exactly."
        ),
    )


class DateFilter(BaseModel):
    """date フィールドに対する範囲フィルタ条件. gte, lte のどちらか一方のみ指定可能."""

    model_config = ConfigDict(validate_by_name=True)
    gte: str | None = Field(
        None, alias="$gte", description="Lower bound (inclusive), 'YYYY-MM-DD HH:MM:SS' format."
    )
    lte: str | None = Field(
        None, alias="$lte", description="Upper bound (inclusive), 'YYYY-MM-DD HH:MM:SS' format."
    )

    @model_validator(mode="after")
    def _check_single_bound(self) -> DateFilter:
        if self.gte is not None and self.lte is not None:
            raise ValueError(
                "gte と lte は同時に指定できません。どちらか一方のみ指定してください。"
            )
        return self


class SearchFilters(BaseModel):
    """obsidian_vault_search が受け付けるメタデータフィルタ条件."""

    model_config = ConfigDict(validate_by_name=True)
    file_path: FileFilter | None = Field(None, description="Vault-relative file path filter.")
    date: DateFilter | None = Field(None, description="Note creation date/time range filter.")


class VaultObsidianRetriever:
    """Obsidian Vault レトリーバー."""

    def __init__(
        self,
        chunk_entity: type,
        store_conn: StoreConnector,
        splitter: TextSplitter,
        embed_model: Embeddings,
        vault_entity: type[base.ObsidianFields],
    ):
        """Construct VaultObsidianRetriever.

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
    async def search_documents(self, query: str, top_k: int, **kwargs: Any) -> Sequence[Document]:
        """チャンク類似検索 → document_id 重複除去して返す.

        返却する Document は id と metadata のみを保持し、page_content は空文字列である。
        全文が必要な場合は get_documents_by_ids を使用すること。
        """
        await self.initialize()
        filters = kwargs.get("filters")
        if isinstance(filters, SearchFilters):
            filters = filters.model_dump(by_alias=True, exclude_none=True) or None

        chunks = await self._vector_store.asimilarity_search(query, k=top_k, filter=filters)

        # document_id 重複除去しつつ Document (id + metadata のみ) を構築
        seen: dict[str, Document] = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id is not None and doc_id not in seen:
                seen[doc_id] = Document(id=doc_id, page_content="", metadata=chunk.metadata)
        return list(seen.values())

    @mlflow.trace(span_type="RETRIEVER")
    async def get_documents_by_ids(self, document_ids: Sequence[str]) -> Sequence[Document]:
        """document_id の完全一致する Document を取得する."""
        await self.initialize()
        async with AsyncSession(self._sa_engine) as session:
            rows = (
                await session.scalars(
                    select(self._vault_entity).where(
                        self._vault_entity.document_id.in_(document_ids)
                    )
                )
            ).all()

        # raw から全文取得
        id_to_doc = {
            row.document_id: Document(
                id=row.document_id,
                page_content=row.content,
                metadata={
                    "document_content_hash": row.document_content_hash,
                    "file_path": row.file_path,
                    **(row.document_metadata or {}),
                },
            )
            for row in rows
        }
        return [id_to_doc[doc_id] for doc_id in document_ids if doc_id in id_to_doc]

    async def get_backlinks(self, document_id: str) -> Sequence[Document]:
        """document_id のノートにリンクしているノートを返す（バックリンク）.

        doc_metadata["forward_links"] は document_id のリストを格納している前提。
        document_id が空文字列の場合は ValueError を raise する。
        """
        await self.initialize()
        if not document_id:
            raise ValueError("document_id must not be empty")
        async with AsyncSession(self._sa_engine) as session:
            rows = (
                await session.scalars(
                    select(self._vault_entity).where(
                        self._vault_entity.backlink_filter(document_id)
                    )
                )
            ).all()
        return [
            Document(
                id=row.document_id,
                page_content=row.content,
                metadata={
                    "document_content_hash": row.document_content_hash,
                    "file_path": row.file_path,
                    **(row.document_metadata or {}),
                },
            )
            for row in rows
        ]

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
                    "document_id": row.document_id,
                    "document_content_hash": row.document_content_hash,
                    "file_path": row.file_path,
                    **{
                        k: v
                        for k, v in (row.document_metadata or {}).items()
                        if k != "forward_links"
                    },
                },
            )
            for row in diff_rows
        ]

        chunks = self._splitter.split_documents(docs)
        await self._vector_store.aadd_documents(chunks)


@ContextRegistry.register("obsidian_retriever")
def build(store_conn: StoreConnector | None = None, **_: Any) -> VaultObsidianRetriever | None:
    """Obsidian レトリーバーを生成する (chunk_size=1024)."""
    if store_conn is None:
        return None
    env_gemini_api_key = os.getenv("ENV_GEMINI_API_KEY")
    assert env_gemini_api_key is not None

    return VaultObsidianRetriever(
        chunk_entity=entities.ObsidianChunkEntity,
        store_conn=store_conn,
        splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200),
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(env_gemini_api_key),
        ),
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
        chunk_entity=entities.ObsidianChunkEntity,
        store_conn=store_conn,
        splitter=RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80),
        embed_model=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=SecretStr(env_gemini_api_key),
        ),
        vault_entity=entities.ObsidianEntity,
    )
