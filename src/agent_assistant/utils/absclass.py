import abc
from typing import Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document


class ChunkReader(abc.ABC):
    """VectorStore による類似検索を行うインターフェース。

    store_name は __init__ で受け取る。
    派生クラスを対象データセット & ドキュメント埋め込み戦略毎に作成する想定。
    """

    def __init__(self, store_name: str):
        self.store_name = store_name

    @abc.abstractmethod
    def get_vectorstore(self) -> VectorStore:
        raise NotImplementedError


class ChunkWriter(abc.ABC):
    """VectorStore へのドキュメント投入を行うインターフェース。

    store_name は __init__ で受け取る。
    バックエンド毎に投入経路が異なる場合 (例: Spark ETL / Delta Sync) に
    VectorReader と分離して実装する。
    """

    def __init__(self, store_name: str):
        self.store_name = store_name

    @abc.abstractmethod
    def add_chunks(self, chunks: list[Document]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def del_chunks(
        self, chunk_ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> None:
        raise NotImplementedError


class DocumentChunker(abc.ABC):
    """ドキュメントをチャンクに分割するインターフェース。

    DocumentRetriever.sync() だけでなく、
    Spark ETL パイプラインの pandas_udf など別の投入経路からも
    直接利用できるよう独立した抽象として定義する。
    """

    @abc.abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class DocumentRetriever(abc.ABC):
    """Document 検索とチャンク同期を行うクラス。"""

    @abc.abstractmethod
    def search_documents(self, query: str, top_k: int) -> list[Document]:
        raise NotImplementedError

    @abc.abstractmethod
    def sync_chunks(self) -> None:
        raise NotImplementedError
