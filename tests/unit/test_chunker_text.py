from langchain_core.documents import Document
from agent_assistant.utils.chunker.text import TextChunker


def test_chunk_01():
    """句点含みテキストが句点で分割される（意味境界）。

    観点1: 2つの文（句点区切り）が chunk_size 内に収まらない場合、句点で分割される
    観点2: 各チャンクの page_content が元テキストの部分文字列になる
    """
    # 試験準備: 2文で構成、各文が chunk_size を超えるテキスト
    text = "人工知能は現代社会で急速に普及しています。医療分野では画像診断に活用されています。"
    doc = Document(page_content=text, metadata={"source": "note.md"})

    # 試験実施: chunk_size を1文の長さ未満に設定
    chunker = TextChunker(chunk_size=20, chunk_overlap=0)
    chunks = chunker.chunk([doc])

    # 結果検証
    # 観点1: 複数チャンクに分割される
    assert len(chunks) >= 2
    # 観点2: 各チャンクが元テキストの部分文字列
    for chunk in chunks:
        assert chunk.page_content in text or text.find(chunk.page_content.rstrip()) >= 0


def test_chunk_02():
    """元のメタデータがすべて元 Document から引き継がれる。

    観点1: source やその他の任意のメタデータがチャンクに保持される
    観点2: 複数ドキュメントでそれぞれのメタデータが正しく引き継がれる
    """
    # 試験準備
    doc1 = Document(
        page_content="aaaa",
        metadata={"source": "folder/note1.md", "author": "Alice"},
    )
    doc2 = Document(
        page_content="bbbb",
        metadata={"source": "note2.md", "tags": ["test"]},
    )

    # 試験実施
    chunker = TextChunker(chunk_size=300)
    chunks = chunker.chunk([doc1, doc2])

    # 結果検証
    # 観点1: メタデータが引き継がれる
    assert chunks[0].metadata["source"] == "folder/note1.md"
    assert chunks[0].metadata["author"] == "Alice"
    # 観点2: 複数ドキュメントで正しく引き継がれる
    assert chunks[1].metadata["source"] == "note2.md"
    assert chunks[1].metadata["tags"] == ["test"]


def test_chunk_03():
    """start_index が各チャンクの正しい開始位置（文字オフセット）になる。

    観点1: 最初のチャンクの start_index は 0
    観点2: 各 start_index が元テキスト内の実際の開始位置と一致する
    """
    # 試験準備
    text = "段落A。段落B。段落C。"
    doc = Document(page_content=text, metadata={"source": "note.md"})

    # 試験実施
    chunker = TextChunker(chunk_size=5, chunk_overlap=0)
    chunks = chunker.chunk([doc])

    # 結果検証
    # 観点1: 最初のチャンクは start_index=0
    assert chunks[0].metadata["start_index"] == 0
    # 観点2: start_index が元テキスト内の正しい位置
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        content = chunk.page_content
        assert text[start : start + len(content)] == content


def test_chunk_04():
    """chunk_size 未満のテキストは 1 チャンクになる。

    観点1: 短いテキストが chunk_size=300 で 1 チャンクになる
    観点2: チャンクの内容が元テキストを含む
    """
    # 試験準備
    doc = Document(page_content="短いテキスト。", metadata={"source": "short.md"})

    # 試験実施
    chunker = TextChunker(chunk_size=300)
    chunks = chunker.chunk([doc])

    # 結果検証
    # 観点1: 1チャンクのみ
    assert len(chunks) == 1
    # 観点2: 元テキストと一致
    assert chunks[0].page_content == "短いテキスト。"


def test_chunk_05():
    """各チャンクが chunk_size 以下の文字数になる。

    観点1: chunk_size を超えるテキストは複数チャンクになる
    観点2: 各チャンクが chunk_size 以下
    """
    # 試験準備: 句点なしの長いテキスト（文字単位分割にフォールバック）
    doc = Document(
        page_content="a" * 50,
        metadata={"source": "note.md"},
    )

    # 試験実施
    chunker = TextChunker(chunk_size=20, chunk_overlap=0)
    chunks = chunker.chunk([doc])

    # 結果検証
    # 観点1: 複数チャンク
    assert len(chunks) > 1
    # 観点2: 各チャンクが chunk_size 以下
    assert all(len(c.page_content) <= 20 for c in chunks)
