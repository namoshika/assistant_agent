import datetime
import pytest
from pathlib import Path
from langchain_community.document_loaders import ObsidianLoader
from agent_assistant.loader.obsidian import VaultLoader
from agent_assistant.retriever.obsidian import path_to_document_id


VAULT_PATH = Path(__file__).parent.parent / "data" / "vault"


@pytest.fixture(scope="module")
def docs():
    """テスト vault から VaultLoader でロードした Document リスト。"""
    return VaultLoader(VAULT_PATH).load()


def test_load_01(docs):
    """VaultLoader.load() が返す Document の metadata が正しく変換される。

    観点1: 全 Document に forward_links が存在する
    観点2: note_a.md の path が vault 相対パスに変換される
    観点3: note_a.md の forward_links が document_id (UUID5) に解決される
    観点4: note_a.md の created / last_modified / last_accessed が ISO 文字列に変換される
    観点5: 全 Document の metadata に "None" 文字列が存在しない
    """
    for doc in docs:
        # 観点1
        assert "forward_links" in doc.metadata
        for value in doc.metadata.values():
            # 観点5
            assert value != "None"

    doc_a = next(d for d in docs if d.metadata.get("path") == "note_a.md")
    # 観点2
    assert not doc_a.metadata["path"].startswith("/")
    # 観点3
    assert doc_a.metadata["forward_links"] == [path_to_document_id("note_b.md")]
    # 観点4
    for key in ("created", "last_modified", "last_accessed"):
        datetime.datetime.fromisoformat(doc_a.metadata[key])


def test_load_02():
    """ObsidianLoader 単体では None フィールドが "None" 文字列になる。
    VaultLoader が後処理で修正している根拠となるバグを確認する。

    観点1: ObsidianLoader は None フロントマター値を "None" 文字列に変換する
    """
    raw_docs = ObsidianLoader(str(VAULT_PATH), collect_metadata=True).load()
    note_c_raw = next(d for d in raw_docs if d.metadata.get("source") == "note_c.md")
    assert note_c_raw.metadata["author"] == "None"
