import datetime
import pytest
from pathlib import Path
from agent_assistant.obsidian import VaultLoader


@pytest.fixture(scope="module")
def docs():
    """テスト vault から VaultLoader でロードした Document リスト。"""
    VAULT_PATH = Path(__file__).parent.parent / "data" / "vault"
    return VaultLoader(VAULT_PATH).load()


def test_load_01(docs):
    """全 Document の metadata に forward_links が存在する。

    観点1: forward_links キーが存在する
    """
    for doc in docs:
        assert "forward_links" in doc.metadata


def test_load_02(docs):
    """note_a.md の path・forward_links・タイムスタンプが正しく変換される。

    観点1: path が vault 相対パスに変換される
    観点2: forward_links が正しく解決される
    観点3: created / last_modified / last_accessed が ISO 文字列に変換される
    """
    doc = next(d for d in docs if d.metadata.get("path") == "note_a.md")
    # 観点1: vault 相対パスになっている（絶対パスでない）
    assert doc.metadata["path"] == "note_a.md"
    assert not doc.metadata["path"].startswith("/")
    # 観点2: wikilink が vault 相対パスに解決されている
    assert doc.metadata["forward_links"] == ["note_b.md"]
    # 観点3: ISO 文字列としてパースできる
    for key in ("created", "last_modified", "last_accessed"):
        datetime.datetime.fromisoformat(doc.metadata[key])
