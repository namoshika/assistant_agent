import abc
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document


class ChunkReader(abc.ABC):
    """VectorStore による類似検索を行うインターフェース。

    派生クラスを対象データセット & ドキュメント埋め込み戦略毎に作成する想定。
    """

    @abc.abstractmethod
    def get_vectorstore(self, store_name: str) -> VectorStore:
        raise NotImplementedError


class ChunkWriter(abc.ABC):
    """VectorStore へのドキュメント投入を行うインターフェース。

    バックエンド毎に投入経路が異なる場合 (例: Spark ETL / Delta Sync) に
    VectorReader と分離して実装する。
    """

    @abc.abstractmethod
    def add_chunks(self, store_name: str, chunks: list[Document]) -> None:
        raise NotImplementedError


class DocumentChunker(abc.ABC):
    """ドキュメントをチャンクに分割するインターフェース。

    DocumentStore.import_documents() だけでなく、
    Spark ETL パイプラインの pandas_udf など別の投入経路からも
    直接利用できるよう独立した抽象として定義する。
    """

    @abc.abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class DocumentStore(abc.ABC):
    """Document 検索とインポートを行うインターフェース。

    派生クラスを対象データセット & チャンキング戦略毎に作成する想定。
    クラスを分け、弄った際にすぐに元の戦略に戻せる状態にすることを推奨。
    """

    @abc.abstractmethod
    def connect(self):
        raise NotImplementedError

    @abc.abstractmethod
    def search_documents(self, query: str, top_k: int) -> list[Document]:
        raise NotImplementedError

    @abc.abstractmethod
    def import_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError
