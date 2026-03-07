from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..absclass import DocumentChunker


class TextChunker(DocumentChunker):
    """RecursiveCharacterTextSplitter ベースの汎用チャンカー。

    意味境界（句読点・改行）を優先してドキュメントを chunk_size 文字以内に分割する。
    各チャンクは元ドキュメントのメタデータを継承し、原文内の開始位置 start_index を追加で記録する。
    """

    _JAPANESE_SEPARATORS = [
        # "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "、",
        "．",
        "，",
        "\u200b",  # ゼロ幅スペース（CJK 用）
        "　",  # 全角スペース
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        separators: list[str] | None = None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            separators=separators or self._JAPANESE_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator="end",
            add_start_index=True,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)
