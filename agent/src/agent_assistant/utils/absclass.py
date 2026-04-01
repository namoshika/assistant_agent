import abc
from typing import Any, Sequence

from langchain_core.documents import Document


class DocumentRetriever(abc.ABC):
    """Document 検索とチャンク同期を行うクラス."""

    @abc.abstractmethod
    def search_documents(self, query: str, top_k: int, **kwargs: Any) -> Sequence[Document]:
        """クエリに基づき、関連するドキュメントを検索する.

        Args:
            query: 検索クエリ文字列。
            top_k: 取得するドキュメントの最大数。
            **kwargs: 実装クラス固有のオプション引数。
                ObsidianLlamaRetriever では filters (MetadataFilters) を受け付ける。

        Returns:
            検索結果の Document リスト。

        """
        raise NotImplementedError

    @abc.abstractmethod
    def sync_chunks(self) -> None:
        """ソースからドキュメントを取得し、チャンクに分割して VectorStore に同期する."""
        raise NotImplementedError
