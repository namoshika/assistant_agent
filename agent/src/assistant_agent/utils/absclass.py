import abc

from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from sqlalchemy import Engine


class StoreConnector(abc.ABC):
    """VectorStore / DocumentStore / SQLAlchemy Engine を生成するファクトリ抽象クラス."""

    @abc.abstractmethod
    def get_engine(self) -> Engine:
        """SQLAlchemy Engine を生成する."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_vector_store(self, name: str, embed_dim: int) -> BasePydanticVectorStore:
        """VectorStore を生成する."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_docstore(self, name: str) -> BaseDocumentStore:
        """DocumentStore を生成する."""
        raise NotImplementedError
