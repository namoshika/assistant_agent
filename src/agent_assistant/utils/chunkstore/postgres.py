from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGEngine, PGVectorStore, Column
from agent_assistant.utils import absclass


class PGVectorChunkStore(absclass.ChunkReader, absclass.ChunkWriter):
    def __init__(
        self,
        engine: PGEngine,
        metadata_columns: list[Column],
        embedding: Embeddings,
        dimention_size: int,
    ):
        self.engine = engine
        self.metadata_columns = metadata_columns
        self.embedding = embedding
        self.dimention_size = dimention_size

    def get_vectorstore(self, store_name: str) -> VectorStore:
        """
        指定されたテーブル名に対応する VectorStore インスタンスを返す。
        テーブルが存在しない場合は初期化してから再取得する。

        Args:
            table_name: ベクトルストアのテーブル名。

        Returns:
            PGVectorStore インスタンス。
        """
        try:
            vectorstore = PGVectorStore.create_sync(
                engine=self.engine,
                table_name=store_name,
                embedding_service=self.embedding,
                metadata_columns=[item.name for item in self.metadata_columns],
            )
        except ValueError as e:
            self._init_table(store_name)
            vectorstore = PGVectorStore.create_sync(
                engine=self.engine,
                table_name=store_name,
                embedding_service=self.embedding,
                metadata_columns=[item.name for item in self.metadata_columns],
            )
        return vectorstore

    def add_chunks(self, store_name: str, documents: list[Document]) -> None:
        """
        ドキュメントをベクトルストアに追加する。

        Args:
            store_name: 追加先のテーブル名。
            documents: 追加する Document のリスト。
        """
        vectorstore = self.get_vectorstore(store_name)
        vectorstore.add_documents(documents)

    def _init_table(self, table_name: str):
        self.engine.init_vectorstore_table(
            table_name,
            self.dimention_size,
            metadata_columns=self.metadata_columns,  # pyright: ignore[reportArgumentType]
        )
