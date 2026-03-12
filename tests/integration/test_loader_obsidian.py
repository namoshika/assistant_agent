import pytest
from pathlib import Path
from agent_assistant.loader.obsidian import VaultLoader


@pytest.fixture(scope="module")
def docs():
    """vault から VaultLoader でロードした Document リスト。"""
    VAULT_PATH = Path("docs/dataset_obsidian/")
    if not VAULT_PATH.exists():
        pytest.fail("Vault が存在しないため失敗")
    return VaultLoader(VAULT_PATH).load()


@pytest.mark.integration
def test_load_01(docs):
    """load() が Document リストを正しく返す。

    観点1: 結果が空でない
    観点2: どの doc も path が絶対パスでなく、空でない
    観点3: forward_links キーが存在しリスト型である
    観点4: forward_links が空でない Document が 1 件以上存在する
    """
    # 結果検証
    # 観点1
    assert len(docs) >= 1
    for doc in docs:
        path = doc.metadata.get("path", "")
        # 観点2
        assert not Path(path).is_absolute(), f"path が絶対パス: {path}"
        assert path != ""
        # 観点3
        assert (
            "forward_links" in doc.metadata
        ), f"forward_links なし: {doc.metadata.get('path')}"
        assert isinstance(doc.metadata["forward_links"], list)
    # 観点4
    has_links = any(len(doc.metadata["forward_links"]) > 0 for doc in docs)
    assert has_links, "forward_links が空でない doc が 1 件もない"
