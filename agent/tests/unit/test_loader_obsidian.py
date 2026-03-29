from pathlib import Path

from langchain_community.document_loaders import ObsidianLoader

from agent_assistant.loader.obsidian import VaultLoader, path_to_document_id

VAULT_PATH = Path(__file__).parent.parent / "data" / "vault"


def test_load_01():
    """VaultLoader.load() が返す Document の metadata が正しく変換される.

    観点1: 全 Document に保証された全メタデータキーが存在し、doc.id が設定されている
    観点2: note_a.md の doc.id が path から生成した document_id (UUID5) と一致する
    観点3: note_a.md の path が vault 相対パスに変換される
    観点4: note_a.md の forward_links が document_id (UUID5) に解決される
    観点5: 除外フィールド (hash / created / last_modified / last_accessed / source) が存在しない
    観点6: 全 Document の metadata に "None" 文字列が存在しない
    """
    # 試験準備 & 試験実施
    docs = VaultLoader(VAULT_PATH).load()

    # 結果検証
    for doc in docs:
        # 観点1
        for key in ("path", "forward_links"):
            assert key in doc.metadata, f"{key} が存在しない: {doc.metadata.get('path')}"
        assert doc.id is not None
        assert not Path(doc.metadata["path"]).is_absolute()
        assert isinstance(doc.metadata["forward_links"], list)
        # 観点5
        for key in ("hash", "created", "last_modified", "last_accessed", "source"):
            assert key not in doc.metadata
        for value in doc.metadata.values():
            # 観点6
            assert value != "None"

    doc_a = next(d for d in docs if d.metadata.get("path") == "note_a.md")
    # 観点2
    assert doc_a.id == path_to_document_id(doc_a.metadata["path"])
    # 観点3
    assert not doc_a.metadata["path"].startswith("/")
    # 観点4
    assert doc_a.metadata["forward_links"] == [path_to_document_id("note_b.md")]


def test_load_02():
    """ObsidianLoader 単体では None フィールドが "None" 文字列になる.

    VaultLoader が後処理で修正している根拠となるバグを確認する
    観点1: ObsidianLoader は None フロントマター値を "None" 文字列に変換する
    """
    # 試験準備 & 試験実施
    raw_docs = ObsidianLoader(VAULT_PATH, collect_metadata=True).load()

    # 結果検証
    note_c_raw = next(d for d in raw_docs if d.metadata.get("source") == "note_c.md")
    assert note_c_raw.metadata["author"] == "None"
