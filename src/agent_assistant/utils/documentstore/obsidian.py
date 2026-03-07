import json
from langchain_core.documents import Document
from sqlalchemy import Engine, text
from typing import Literal

from .. import absclass
from ..retriever import ObsidianChunkStore
from ..chunker.text import TextChunker


class ObsidianDocumentStore(absclass.DocumentStore):
    """Obsidian Vault のノートを検索・取得する DocumentStore。

    PostgreSQL 上に2テーブルを管理する:
    - {store_name}_raw   : ノート原文 (plain SQL)
    - {store_name}_chunks: 文字数チャンク + 埋め込み (PGVectorStore)

    search_documents() はチャンク類似検索 → path 重複除去 → 原文取得 のパイプライン。
    get_documents()    は path (ファイル名 / 相対パス) の後方一致で原文を直接取得する。
    """

    def __init__(
        self,
        store_name: str,
        chunk_store: ObsidianChunkStore,
        sa_engine: Engine,
        chunker: TextChunker,
    ):
        self._store_name = store_name
        self._chunk_store = chunk_store
        self._sa_engine = sa_engine
        self._chunker = chunker
        self._chunk_vs = None

    @property
    def _chunk_table(self) -> str:
        return f"{self._store_name}_chunks"

    @property
    def _raw_table(self) -> str:
        return f"{self._store_name}_raw"

    def connect(self) -> None:
        if self._chunk_vs is not None:
            return
        # raw テーブルを作成 (存在する場合はスキップ)
        with self._sa_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self._raw_table} "
                    f"(path TEXT PRIMARY KEY, content TEXT NOT NULL, metadata JSONB)"
                )
            )
            conn.commit()
        # chunk テーブルを初期化 (自動テーブル作成付き)
        self._chunk_vs = self._chunk_store.get_vectorstore(self._chunk_table)

    def import_documents(self, documents: list[Document]) -> None:
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")

        paths = [doc.metadata.get("path", "") for doc in documents]

        with self._sa_engine.connect() as conn:
            # raw テーブルへ upsert
            for doc in documents:
                conn.execute(
                    text(
                        f"INSERT INTO {self._raw_table} (path, content, metadata) "
                        f"VALUES (:path, :content, :metadata) "
                        f"ON CONFLICT (path) DO UPDATE "
                        f"SET content = EXCLUDED.content, metadata = EXCLUDED.metadata"
                    ),
                    {
                        "path": doc.metadata.get("path", ""),
                        "content": doc.page_content,
                        "metadata": json.dumps(doc.metadata),
                    },
                )
            # 既存チャンクを削除
            conn.execute(
                text(f"DELETE FROM {self._chunk_table} WHERE path = ANY(:paths)"),
                {"paths": paths},
            )
            conn.commit()

        # 新チャンクを追加
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

        # search() はスコア降順でチャンクを返すため、各 path の初出 = 最高スコアチャンク
        chunks = self._chunk_vs.search(query, search_type, **kwargs)
        sorted_paths = list(dict.fromkeys(c.metadata["path"] for c in chunks))

        # 原文取得 → sorted_paths 順に並び替えて返す (SQL ANY は行順を保証しないため)
        docs = self._fetch_raw(sorted_paths)
        path_to_doc = {doc.metadata["path"]: doc for doc in docs}
        return [path_to_doc[p] for p in sorted_paths if p in path_to_doc]

    def get_documents(self, document_ids: list[str]) -> list[Document]:
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")

        return self._fetch_by_suffix(document_ids)

    def get_backlinks(self, paths: list[str]) -> list[Document]:
        """paths のノートにリンクしているノートを返す（バックリンク）。

        将来 GIN インデックス対応時は @> クエリへの切り替えで高速化可能。
        """
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")
        if not paths:
            return []
        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT path, content, metadata FROM {self._raw_table} "
                    f"WHERE EXISTS ("
                    f"  SELECT 1 FROM jsonb_array_elements_text(metadata->'forward_links') fl "
                    f"  WHERE fl = ANY(:paths)"
                    f")"
                ),
                {"paths": paths},
            )
            return [
                Document(
                    page_content=row.content,
                    metadata=(
                        json.loads(row.metadata)
                        if isinstance(row.metadata, str)
                        else (row.metadata or {})
                    ),
                )
                for row in result
            ]

    def _fetch_raw(self, paths: list[str]) -> list[Document]:
        """path の完全一致リストで raw テーブルから Document を取得する。"""
        if not paths:
            return []
        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT path, content, metadata FROM {self._raw_table} "
                    f"WHERE path = ANY(:paths)"
                ),
                {"paths": paths},
            )
            return [
                Document(
                    page_content=row.content,
                    metadata=(
                        json.loads(row.metadata)
                        if isinstance(row.metadata, str)
                        else (row.metadata or {})
                    ),
                )
                for row in result
            ]

    def get_backlinks(self, paths: list[str]) -> list[Document]:
        """paths のノートにリンクしているノートを返す（バックリンク）。

        将来 GIN インデックス対応時は @> クエリへの切り替えで高速化可能。
        """
        if self._chunk_vs is None:
            raise ValueError("Store is not connected")
        if not paths:
            return []
        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT path, content, metadata FROM {self._raw_table} "
                    f"WHERE EXISTS ("
                    f"  SELECT 1 FROM jsonb_array_elements_text(metadata->'forward_links') fl "
                    f"  WHERE fl = ANY(:paths)"
                    f")"
                ),
                {"paths": paths},
            )
            return [
                Document(
                    page_content=row.content,
                    metadata=(
                        json.loads(row.metadata)
                        if isinstance(row.metadata, str)
                        else (row.metadata or {})
                    ),
                )
                for row in result
            ]

    def _fetch_by_suffix(self, document_ids: list[str]) -> list[Document]:
        """document_ids の後方一致 (ファイル名 or フルパス) で raw テーブルから Document を取得する。

        各 id について: path = :id OR path LIKE '%/' || :id
        """
        if not document_ids:
            return []

        conditions = " OR ".join(
            f"path = :id_{i} OR path LIKE '%/' || :id_{i}"
            for i in range(len(document_ids))
        )
        params = {f"id_{i}": doc_id for i, doc_id in enumerate(document_ids)}

        with self._sa_engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT path, content, metadata FROM {self._raw_table} "
                    f"WHERE {conditions}"
                ),
                params,
            )
            return [
                Document(
                    page_content=row.content,
                    metadata=(
                        json.loads(row.metadata)
                        if isinstance(row.metadata, str)
                        else (row.metadata or {})
                    ),
                )
                for row in result
            ]
