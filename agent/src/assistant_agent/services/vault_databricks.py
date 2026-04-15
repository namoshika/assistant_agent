import os
from typing import Any, Sequence

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksVectorSearch
from langchain_core.documents import Document

from assistant_agent.utils.context import ContextRegistry


class VaultDatabricksRetriever:
    """Databricks Vector Search レトリーバー."""

    def __init__(self, index_name: str):
        """Construct VaultDatabricksRetriever.

        Args:
            index_name: Delta Sync Index 名。"catalog.schema.name" 形式。

        """
        self._vector_store = DatabricksVectorSearch(
            index_name=index_name,
            workspace_client=WorkspaceClient(),
        )

    @mlflow.trace(span_type="RETRIEVER")
    def search_documents(self, query: str, top_k: int) -> Sequence[Document]:
        """チャンク類似検索."""
        return self._vector_store.similarity_search(query=query, k=top_k)


@ContextRegistry.register("sample_retriever", variant="qwen3emb06b")
def build_qwen3emb06b(**_: Any) -> VaultDatabricksRetriever:
    """Databricks レトリーバーを生成する."""
    index_name = os.environ["ENV_DATABRICKS_VSI_WEBSITE"]
    return VaultDatabricksRetriever(index_name=index_name)
