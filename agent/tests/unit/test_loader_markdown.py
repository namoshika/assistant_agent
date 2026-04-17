import tempfile
import uuid
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from assistant_agent.loaders.markdown import MarkdownReader

VAULT_PATH = Path(__file__).parent.parent / "data" / "vault"


def test_load_data_01():
    """MarkdownReader.lazy_load_data() がフロントマターありの場合に正しく動作するか確認.

    観点1: text がフロントマターを除いた本文のみであること
    観点2: フロントマターのキーと値が Document.metadata に取り込まれること。
           datetime 型の値は ISO 文字列に変換され、手動で渡した extra_info のキーも含まれること
    観点3: lazy_load_data() の結果と load_data() の結果が一致すること
    観点4: SimpleDirectoryReader の file_extractor として使用した際、
           .md ファイル全件が取得され、file_path / file_name 等が Document.metadata に存在すること
    観点5: Document.id_ が extra_info["file_path"] から生成した UUID5 と一致すること
    """
    reader = MarkdownReader()
    note_a = VAULT_PATH / "note_a.md"

    # 試験実施
    docs = list(reader.lazy_load_data(note_a, extra_info={"custom_key": "custom_value"}))

    # 結果検証
    assert len(docs) == 1
    doc = docs[0]

    # 観点1
    assert doc.text == "ノート A の本文。\n\n[[note_b]] へのリンク。\n"
    assert "---" not in doc.text

    # 観点2
    assert (
        doc.metadata["URL"]
        == "https://www.databricks.com/jp/blog/introducing-databricks-free-edition"
    )
    assert doc.metadata["date"] == "2025-06-17T10:10:43"
    assert doc.metadata["tags"] is None
    assert doc.metadata["custom_key"] == "custom_value"

    # 観点3
    docs_via_load = reader.load_data(note_a, extra_info={"custom_key": "custom_value"})
    assert docs[0].text == docs_via_load[0].text
    assert docs[0].metadata == docs_via_load[0].metadata

    # 観点4
    dir_reader = SimpleDirectoryReader(
        input_dir=str(VAULT_PATH),
        file_extractor={".md": reader},
        required_exts=[".md"],
    )
    all_docs = dir_reader.load_data()
    assert len(all_docs) == 3
    for d in all_docs:
        assert "file_path" in d.metadata
        assert "file_name" in d.metadata

    # 観点5
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(note_a.resolve())))
    assert doc.id_ == expected_id


def test_load_data_02():
    """MarkdownReader.lazy_load_data() がフロントマターなしの場合に正しく動作するか確認.

    観点1: text がファイル全文であること
    観点2: Document.metadata が extra_info のみであること
    """
    reader = MarkdownReader()

    # 試験準備: フロントマターなしのコンテンツを tmp ファイルで用意
    body = "フロントマターなしの本文。\n"
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write(body)
        tmp_path = Path(f.name)

    try:
        # 試験実施
        docs = list(reader.lazy_load_data(tmp_path, extra_info={"source": "test"}))

        # 結果検証
        assert len(docs) == 1
        doc = docs[0]

        # 観点1
        assert doc.text == body

        # 観点2
        assert doc.metadata == {"source": "test"}
    finally:
        tmp_path.unlink()
