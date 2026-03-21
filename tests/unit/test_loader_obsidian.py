import datetime
from pathlib import Path

from langchain_community.document_loaders import ObsidianLoader

from agent_assistant.loader.obsidian import VaultLoader, path_to_document_id

VAULT_PATH = Path(__file__).parent.parent / "data" / "vault"


def test_load_01():
    """VaultLoader.load() が返す Document の metadata が正しく変換される.

    観点1: 全 Document に保証された全メタデータキーが存在し、doc.id が設定されている
    観点2: note_a.md の path が document_id (UUID5) に変換される
    観点3: note_a.md の path が vault 相対パスに変換される
    観点4: note_a.md の forward_links が document_id (UUID5) に解決される
    観点5: note_a.md の created / last_modified / last_accessed が ISO 文字列に変換される
    観点6: 全 Document の metadata に "None" 文字列が存在しない
    """
    # 試験準備 & 試験実施
    docs = VaultLoader(VAULT_PATH).load()

    # 結果検証
    for doc in docs:
        # 観点1
        for key in (
            "path",
            "hash",
            "document_id",
            "created",
            "last_modified",
            "last_accessed",
            "forward_links",
        ):
            assert key in doc.metadata, f"{key} が存在しない: {doc.metadata.get('path')}"
        assert doc.id is not None
        assert doc.id == doc.metadata["document_id"]
        assert not Path(doc.metadata["path"]).is_absolute()
        for dt_key in ("created", "last_modified", "last_accessed"):
            datetime.datetime.fromisoformat(doc.metadata[dt_key])
        assert isinstance(doc.metadata["forward_links"], list)
        for value in doc.metadata.values():
            # 観点6
            assert value != "None"

    doc_a = next(d for d in docs if d.metadata.get("path") == "note_a.md")
    # 観点2
    assert doc_a.id == path_to_document_id(doc_a.metadata["path"])
    assert doc_a.metadata["document_id"] == doc_a.id
    # 観点3
    assert not doc_a.metadata["path"].startswith("/")
    # 観点4
    assert doc_a.metadata["forward_links"] == [path_to_document_id("note_b.md")]
    # 観点5
    for key in ("created", "last_modified", "last_accessed"):
        datetime.datetime.fromisoformat(doc_a.metadata[key])


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
