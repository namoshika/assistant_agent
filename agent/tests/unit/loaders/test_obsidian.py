from pathlib import Path

from assistant_agent.loaders.obsidian import ObsidianLoader, path_to_document_id

VAULT_PATH = Path(__file__).parent.parent.parent / "data" / "vault"


def test_load_01():
    """ObsidianReader.load() が返す Document の metadata が正しく変換されるか確認.

    観点1: 全 Document に保証された全メタデータキーが存在し、doc.id が設定されている
    観点2: note_a.md の doc.id が path から生成した document_id (UUID5) と一致する
    観点3: note_a.md の path が vault 相対パスに変換される
    観点4: note_a.md の forward_links が document_id (UUID5) に解決される
    観点5: 除外フィールド (hash / created / last_modified / last_accessed / source) が存在しない
    観点6: note_c.md のフロントマター値が正しく変換される
      - date (datetime) が ISO 文字列に変換され、"None" 文字列にならない
      - URL (str) がそのまま保持される
      - tags (null) が None のまま保持される
    """
    # 試験準備 & 試験実施
    docs = ObsidianLoader(VAULT_PATH).load()

    # 結果検証
    for doc in docs:
        # 観点1
        for key in ("file_path", "forward_links"):
            assert key in doc.metadata, f"{key} が存在しない: {doc.metadata.get('path')}"
        assert doc.id is not None
        assert not Path(doc.metadata["file_path"]).is_absolute()
        assert isinstance(doc.metadata["forward_links"], list)
        # 観点5
        for key in ("hash", "created", "last_modified", "last_accessed", "source"):
            assert key not in doc.metadata

    doc_a = next(d for d in docs if d.metadata.get("file_path") == "note_a.md")
    # 観点2
    assert doc_a.id == path_to_document_id(doc_a.metadata["file_path"])
    # 観点3
    assert not doc_a.metadata["file_path"].startswith("/")
    # 観点4
    assert doc_a.metadata["forward_links"] == [path_to_document_id("note_b.md")]

    doc_c = next(d for d in docs if d.metadata.get("file_path") == "note_c.md")
    # 観点6
    assert doc_c.metadata["date"] == "2025-06-17T10:10:43"
    assert (
        doc_c.metadata["URL"]
        == "https://www.databricks.com/jp/blog/introducing-databricks-free-edition"
    )
    assert doc_c.metadata["tags"] is None
