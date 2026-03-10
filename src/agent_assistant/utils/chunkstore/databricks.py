import pandas as pd
from typing import Callable
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from databricks_langchain import DatabricksVectorSearch
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from agent_assistant.utils import absclass


class DatabricksChunkReader(absclass.ChunkReader):
    def __init__(
        self,
        embedding: Embeddings,
        dimention_size: int,
        text_col: str,
        endpoint_name: str,
    ):
        self.text_col = text_col
        self.embedding = embedding
        self.dimention_size = dimention_size
        self.endpoint_name = endpoint_name

    def get_vectorstore(self, store_name: str) -> VectorStore:
        return DatabricksVectorSearch(
            store_name, self.endpoint_name, self.embedding, self.text_col
        )


def create_vsi(
    index_name: str,
    index_col: str,
    emb_col: str,
    emb_dimsize: int,
    source_name: str,
    endpoint_name: str = "vsi_endpoint",
):
    from databricks.vector_search.client import VectorSearchClient

    client = VectorSearchClient()
    # ---------------------------------------------
    # Vector Search Index 用のエンドポイントを取得
    # ---------------------------------------------
    try:
        res_enp = client.get_endpoint(endpoint_name)
    # 無い場合は作成
    except Exception as ex:
        res_enp = client.create_endpoint_and_wait(
            name=endpoint_name, endpoint_type="STANDARD"
        )

    # ---------------------------------------------
    # Vector Search Index を取得
    # ---------------------------------------------
    try:
        res_vsi = client.get_index(endpoint_name, index_name)
    # 無い場合は作成
    except Exception as ex:
        res_vsi = client.create_delta_sync_index_and_wait(
            endpoint_name,
            index_name,
            index_col,
            source_name,
            "TRIGGERED",
            emb_dimsize,
            emb_col,
            # 一旦全カラムをインデックスへ収録。必要に応じて削るのは派生クラスの役割とする
            # columns_to_sync=["file_name", "content"]
        )
    return res_enp, res_vsi


def emb_udf_factory(factory: Callable[[], Embeddings]):
    """
    Spark DataFrame の文字列列から埋め込み表現列を生成
    """

    @pandas_udf(ArrayType(FloatType()))
    def emb_udf(text_col: pd.Series) -> pd.Series:
        # return pd.Series([[float(text_col.size)] for _ in text_col])
        embedding = factory()
        doc_embed = pd.Series(
            [embbed for embbed in embedding.embed_documents(text_col.tolist())]
        )
        return doc_embed

    return emb_udf
