from langchain_core.documents import Document
from agent_assistant.utils.chunker.markdown import MarkdownHeaderChunker


def test_chunk_01():
    """H1見出しでドキュメントが分割され、source がファイル名のみになる。

    観点1: H1 ヘッダーの数だけチャンクに分割される（H1なしは1チャンクのまま）
    観点2: source メタデータがフルパスからファイル名のみに変換される
    観点3: H1 を持つチャンクには section メタデータが付与される
    """
    # 試験準備
    doc1 = Document(
        page_content="# 月次レポート\n概要テキスト\n# 詳細分析\n詳細テキスト",
        metadata={"source": "/path/to/news/2024-01.md"},
    )
    doc2 = Document(
        page_content="H1見出しなしのプレーンテキスト",
        metadata={"source": "/path/to/news/2024-02.md"},
    )

    # 試験実施
    chunker = MarkdownHeaderChunker()
    chunks = chunker.chunk([doc1, doc2])

    # 観点1: doc1(2セクション) + doc2(H1なし→1チャンク) = 3チャンク
    assert len(chunks) == 3
    # 観点2,3: チャンクごとに本文・source・section を検証
    chunk = chunks[0]
    assert chunk.page_content == "概要テキスト"
    assert chunk.metadata["source"] == "2024-01.md"
    assert chunk.metadata["section"] == "月次レポート"
    chunk = chunks[1]
    assert chunk.page_content == "詳細テキスト"
    assert chunk.metadata["source"] == "2024-01.md"
    assert chunk.metadata["section"] == "詳細分析"
    chunk = chunks[2]
    assert chunk.page_content == "H1見出しなしのプレーンテキスト"
    assert chunk.metadata["source"] == "2024-02.md"
    assert "section" not in chunk.metadata
