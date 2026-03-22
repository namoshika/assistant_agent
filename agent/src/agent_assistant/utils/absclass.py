import abc
from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class ChunkReader(abc.ABC):
    """VectorStore による類似検索を行うインターフェース.

    store_name は __init__ で受け取る。
    派生クラスを対象データセット & ドキュメント埋め込み戦略毎に作成する想定。
    """

    def __init__(self, store_name: str):
        """Construct ChunkReader.

        Args:
            store_name: ベクトルストアの識別名。

        """
        self.store_name = store_name

    @abc.abstractmethod
    def get_vectorstore(self) -> VectorStore:
        """store_name に対応する VectorStore インスタンスを返す.

        Returns:
            VectorStore インスタンス。

        """
        raise NotImplementedError


class ChunkWriter(abc.ABC):
    """VectorStore へのドキュメント投入を行うインターフェース.

    store_name は __init__ で受け取る。
    バックエンド毎に投入経路が異なる場合 (例: Spark ETL / Delta Sync) に
    VectorReader と分離して実装する。
    """

    def __init__(self, store_name: str):
        """Construct ChunkWriter.

        Args:
            store_name: ベクトルストアの識別名。

        """
        self.store_name = store_name

    @abc.abstractmethod
    def add_chunks(self, chunks: list[Document]) -> None:
        """ドキュメントを store_name のベクトルストアに追加する.

        Args:
            chunks: 追加する Document のリスト。

        """
        raise NotImplementedError

    @abc.abstractmethod
    def del_chunks(
        self, chunk_ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> None:
        """ドキュメントを store_name のベクトルストアから削除する.

        Args:
            chunk_ids: 削除する Document.id のリスト。
            filter: メタデータフィルタ。

        """
        raise NotImplementedError


class DocumentChunker(abc.ABC):
    """ドキュメントをチャンクに分割するインターフェース.

    DocumentRetriever.sync() だけでなく、
    Spark ETL パイプラインの pandas_udf など別の投入経路からも
    直接利用できるよう独立した抽象として定義する。
    """

    @abc.abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """ドキュメントをチャンクに分割する.

        Args:
            documents: 分割対象の Document リスト。

        Returns:
            分割後の Document リスト。

        """
        raise NotImplementedError


class DocumentRetriever(abc.ABC):
    """Document 検索とチャンク同期を行うクラス."""

    @abc.abstractmethod
    def search_documents(self, query: str, top_k: int) -> list[Document]:
        """クエリに基づき、関連するドキュメントを検索する.

        Args:
            query: 検索クエリ文字列。
            top_k: 取得するドキュメントの最大数。

        Returns:
            検索結果の Document リスト。

        """
        raise NotImplementedError

    @abc.abstractmethod
    def sync_chunks(self) -> None:
        """ソースからドキュメントを取得し、チャンクに分割して VectorStore に同期する."""
        raise NotImplementedError
