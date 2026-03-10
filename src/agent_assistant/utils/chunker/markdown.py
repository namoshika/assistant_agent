from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from agent_assistant.utils import absclass


class MarkdownHeaderChunker(absclass.DocumentChunker):
    """Markdown の H1 見出し (#) でドキュメントをチャンクに分割する。

    MarkdownDocumentStore.import_documents() からだけでなく、
    Spark pandas_udf など別の投入経路から直接利用できる。

    Spark ETL での利用例:
        chunker = MarkdownHeaderChunker()

        @pandas_udf(ArrayType(StringType()))
        def chunk_udf(text_col: pd.Series, source_col: pd.Series) -> pd.Series:
            return pd.Series([
                [c.page_content for c in chunker.chunk([Document(page_content=t, metadata={"source": s})])]
                for t, s in zip(text_col, source_col)
            ])
    """

    def chunk(self, documents: list[Document]) -> list[Document]:
        docs_chunked = []
        doc_splitter = MarkdownHeaderTextSplitter([("#", "section")])
        for doc_origin in documents:
            file_name = Path(doc_origin.metadata["source"]).name
            doc_splitted = doc_splitter.split_text(doc_origin.page_content)
            for item in doc_splitted:
                item.metadata["source"] = file_name
                docs_chunked.append(item)
        return docs_chunked
