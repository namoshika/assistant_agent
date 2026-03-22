from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import Column, PGEngine, PGVectorStore
from sqlalchemy.orm import DeclarativeBase

from agent_assistant.utils import absclass


class PGVectorChunkStore[CEntity](absclass.ChunkReader, absclass.ChunkWriter):
    def __init__(
        self,
        engine: PGEngine,
        store_name: str,
        metadata_columns: list[Column],
        embedding: Embeddings,
        dimention_size: int,
        chunk_entity: type[CEntity],
        chunk_base: type[DeclarativeBase],
    ):
        """Construct PGVectorChunkStore.

        Args:
            engine: PostgreSQL 接続を管理する PGEngine インスタンス
            store_name: VectorStore が使うテーブル名
            metadata_columns: Document.metadata からテーブル列に展開する項目リスト
            embedding: テキストのベクトル化に使用する Embeddings インスタンス
            dimention_size: 埋め込みベクトルの次元数
            chunk_entity: VectorStore が生成するテーブルの SQLAlchemy 側の定義 (SQLAlchemy から操作)
            chunk_base: SQLAlchemy の DeclarativeBase クラス (テーブル共通項目 (schema など) を設定)

        """
        super().__init__(store_name)
        self.engine = engine
        self.metadata_columns = metadata_columns
        self.embedding = embedding
        self.dimention_size = dimention_size
        self.chunk_entity: type[CEntity] = type(
            f"PGVectorChunkStore_{store_name}",
            (chunk_base, chunk_entity),
            {"__tablename__": store_name, "__table_args__": {"extend_existing": True}},
        )  # pyright: ignore[reportAttributeAccessIssue]

    def get_vectorstore(self) -> VectorStore:
        """store_name に対応する VectorStore インスタンスを返す.

        テーブルが存在しない場合は初期化してから再取得する。

        Returns:
            PGVectorStore インスタンス。

        """
        try:
            vectorstore = PGVectorStore.create_sync(
                engine=self.engine,
                embedding_service=self.embedding,
                table_name=self.store_name,
                metadata_columns=[item.name for item in self.metadata_columns],
            )
        except ValueError:
            self._init_table(self.store_name)
            vectorstore = PGVectorStore.create_sync(
                engine=self.engine,
                embedding_service=self.embedding,
                table_name=self.store_name,
                metadata_columns=[item.name for item in self.metadata_columns],
            )
        return vectorstore

    def add_chunks(self, chunks: list[Document]) -> None:
        """ドキュメントを store_name のベクトルストアに追加する.

        Args:
            chunks: 追加する Document のリスト。

        """
        vectorstore = self.get_vectorstore()
        vectorstore.add_documents(chunks)

    def del_chunks(
        self, chunk_ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> None:
        """ドキュメントを store_name のベクトルストアから削除する.

        Args:
            chunk_ids: 削除する Document.id のリスト。
            filter: メタデータフィルタ。

        """
        vectorstore = self.get_vectorstore()
        vectorstore.delete(chunk_ids, filter=filter)

    def _init_table(self, table_name: str):
        self.engine.init_vectorstore_table(
            table_name,
            self.dimention_size,
            metadata_columns=self.metadata_columns,  # pyright: ignore[reportArgumentType]
        )
