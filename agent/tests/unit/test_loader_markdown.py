import tempfile
import uuid
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader

from assistant_agent.loaders.markdown import MarkdownLoader

VAULT_PATH = Path(__file__).parent.parent / "data" / "vault"


def test_load_data_01():
    """MarkdownLoader.lazy_load() がフロントマターありの場合に正しく動作するか確認.

    観点1: page_content がフロントマターを除いた本文のみであること
    観点2: フロントマターのキーと値が Document.metadata に取り込まれること。
           datetime 型の値は ISO 文字列に変換され、
           コンストラクタに渡した追加メタデータのキーも含まれること
    観点3: lazy_load() の結果と load() の結果が一致すること
    観点4: DirectoryLoader の loader_cls として使用した際、
           .md ファイル全件が取得され、file_path / file_name が Document.metadata に存在すること
    観点5: Document.id が file_path (絶対パス) から生成した UUID5 と一致すること
    """
    note_a = VAULT_PATH / "note_a.md"
    reader = MarkdownLoader(note_a, custom_key="custom_value")

    # 試験実施
    docs = list(reader.lazy_load())

    # 結果検証
    assert len(docs) == 1
    doc = docs[0]

    # 観点1
    assert doc.page_content == "ノート A の本文。\n\n[[note_b]] へのリンク。\n"
    assert "---" not in doc.page_content

    # 観点2
    assert (
        doc.metadata["URL"]
        == "https://www.databricks.com/jp/blog/introducing-databricks-free-edition"
    )
    assert doc.metadata["date"] == "2025-06-17T10:10:43"
    assert doc.metadata["tags"] is None
    assert doc.metadata["custom_key"] == "custom_value"

    # 観点3
    docs_via_load = reader.load()
    assert docs[0].page_content == docs_via_load[0].page_content
    assert docs[0].metadata == docs_via_load[0].metadata

    # 観点4
    dir_reader = DirectoryLoader(
        str(VAULT_PATH),
        glob="*.md",
        loader_cls=MarkdownLoader,  # type: ignore[arg-type]
    )
    all_docs = dir_reader.load()
    assert len(all_docs) == 3
    for d in all_docs:
        assert "file_path" in d.metadata
        assert "file_name" in d.metadata

    # 観点5
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(note_a.resolve())))
    assert doc.id == expected_id


def test_load_data_02():
    """MarkdownLoader.lazy_load() がフロントマターなしの場合に正しく動作するか確認.

    観点1: page_content がファイル全文であること
    観点2: Document.metadata が file_path / file_name のみであること
    """
    # 試験準備: フロントマターなしのコンテンツを tmp ファイルで用意
    body = "フロントマターなしの本文。\n"
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write(body)
        tmp_path = Path(f.name)

    try:
        reader = MarkdownLoader(tmp_path)

        # 試験実施
        docs = list(reader.lazy_load())

        # 結果検証
        assert len(docs) == 1
        doc = docs[0]

        # 観点1
        assert doc.page_content == body

        # 観点2
        assert doc.metadata == {
            "file_path": str(tmp_path.resolve()),
            "file_name": tmp_path.resolve().name,
        }
    finally:
        tmp_path.unlink()
