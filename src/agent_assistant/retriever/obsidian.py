import json
import uuid
from typing import Literal
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, Column
from sqlalchemy import Engine, text
from pydantic import SecretStr

from agent_assistant.utils.chunker.text import TextChunker
from agent_assistant.utils.chunkstore.postgres import PGVectorChunkStore
from agent_assistant.utils import absclass


def path_to_document_id(path: str) -> str:
    """vault 相対パスから document_id (UUID5) を生成する。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


class ObsidianChunkStore(PGVectorChunkStore):
    """Obsidian Vault のチャンクを PGVector に格納するストア。

    各チャンクに元ノートの document_id・path (vault 相対パス)・
    原文内の開始位置 start_index を保持する。
    """

    def __init__(self, engine: PGEngine, emb_api_key: SecretStr):
        super().__init__(
            engine,
            [
                Column("document_id", "text", False),
                Column("path", "text", False),
                Column("start_index", "integer", False),
            ],
            GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", api_key=emb_api_key
            ),
            3072,
        )


class ObsidianDocumentStore(absclass.DocumentStore):
    """Obsidian Vault のノートを検索・取得する DocumentStore。

    PostgreSQL 上に2テーブルを管理する:
    - {store_name}_raw   : ノート原文 (plain SQL)
    - {store_name}_chunks: 文字数チャンク + 埋め込み (PGVectorStore)

    search_documents() はチャンク類似検索 → document_id 重複除去 → 原文取得 のパイプライン。
    get_documents_by_ids()  は document_id の完全一致で原文を直接取得する。
    get_document_by_path()  は path (ファイル名 / 相対パス) の後方一致で原文を直接取得する。
    """

    def __init__(
        self,
        store_name: str,
        chunk_store: ObsidianChunkStore,
        sa_engine: Engine,
        chunker: TextChunker,
    ):
        self._chunk_table = f"{store_name}_chunks"
        self._raw_table = f"{store_name}_raw"
        self._chunk_store = chunk_store
        self._sa_engine = sa_engine
        self._chunker = chunker
        self._chunk_vs = None

    def connect(self) -> None:
        if self._chunk_vs is not None:
            return
        # raw テーブルを作成 (存在する場合はスキップ)
        with self._sa_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self._raw_table} "
                    f"(document_id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, "
                    f"content TEXT NOT NULL, metadata JSONB)"
                )
            )
            conn.commit()
        # chunk テーブルを初期化 (自動テーブル作成付き)
        self._chunk_vs = self._chunk_store.get_vectorstore(self._chunk_table)

    def import_documents(self, documents: list[Document]) -> None:
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")

        # document_id を付与 (chunker.chunk() が継承できるよう事前に設定)
        for doc in documents:
            doc.metadata["document_id"] = path_to_document_id(
                doc.metadata.get("path", "")
            )
        doc_ids = [doc.metadata["document_id"] for doc in documents]

        with self._sa_engine.connect() as conn:
            # raw テーブルへ upsert
            for doc in documents:
                conn.execute(
                    text(
                        f"INSERT INTO {self._raw_table} "
                        f"(document_id, path, content, metadata) "
                        f"VALUES (:document_id, :path, :content, :metadata) "
                        f"ON CONFLICT (path) DO UPDATE "
                        f"SET document_id = EXCLUDED.document_id, "
                        f"content = EXCLUDED.content, metadata = EXCLUDED.metadata"
                    ),
                    {
                        "document_id": doc.metadata["document_id"],
                        "path": doc.metadata.get("path", ""),
                        "content": doc.page_content,
                        "metadata": json.dumps(doc.metadata),
                    },
                )
            # 既存チャンクを削除
            conn.execute(
                text(
                    f"DELETE FROM {self._chunk_table} WHERE document_id = ANY(:doc_ids)"
                ),
                {"doc_ids": doc_ids},
            )
            conn.commit()

        # 新チャンクを追加 (split_documents が document_id を継承する)
        chunks = self._chunker.chunk(documents)
        self._chunk_store.add_chunks(self._chunk_table, chunks)

    def search_documents(
        self,
        query: str,
        search_type: Literal[
            "similarity", "mmr", "similarity_score_threshold"
        ] = "similarity",
        **kwargs,
    ) -> list[Document]:
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")

        # search() はスコア降順でチャンクを返すため、各 document_id の初出 = 最高スコアチャンク
        chunks = self._chunk_vs.search(query, search_type, **kwargs)
        sorted_ids = list(dict.fromkeys(c.metadata["document_id"] for c in chunks))

        # 原文取得 → sorted_ids 順に並び替えて返す
        docs = self.get_documents_by_ids(sorted_ids)
        id_to_doc = {doc.metadata["document_id"]: doc for doc in docs}
        return [id_to_doc[doc_id] for doc_id in sorted_ids if doc_id in id_to_doc]

    def get_documents_by_ids(self, document_ids: list[str]) -> list[Document]:
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")

        if not document_ids:
            return []
        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT document_id, path, content, metadata FROM {self._raw_table} "
                    f"WHERE document_id = ANY(:ids)"
                ),
                {"ids": document_ids},
            )
            return [self._row_to_doc(row) for row in result]

    def get_document_by_path(self, path: str) -> list[Document]:
        """path の後方一致 (ファイル名 or フルパス) で raw テーブルから Document を取得する。"""
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")

        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT document_id, path, content, metadata FROM {self._raw_table} "
                    f"WHERE path = :path OR path LIKE '%/' || :path"
                ),
                {"path": path},
            )
            return [self._row_to_doc(row) for row in result]

    def get_backlinks(self, document_id: str) -> list[Document]:
        """document_id のノートにリンクしているノートを返す（バックリンク）。

        forward_links は document_id のリストを格納している前提。
        """
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")
        if not document_id:
            raise ValueError("document_id must not be empty")
        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT document_id, path, content, metadata FROM {self._raw_table} "
                    f"WHERE EXISTS ("
                    f"  SELECT 1 FROM jsonb_array_elements_text(metadata->'forward_links') fl "
                    f"  WHERE fl = :document_id"
                    f")"
                ),
                {"document_id": document_id},
            )
            return [self._row_to_doc(row) for row in result]

    def _row_to_doc(self, row) -> Document:
        """DB 行から Document を生成し、document_id を metadata に注入する。"""
        meta = (
            json.loads(row.metadata)
            if isinstance(row.metadata, str)
            else (row.metadata or {})
        )
        meta["document_id"] = row.document_id
        return Document(page_content=row.content, metadata=meta)
