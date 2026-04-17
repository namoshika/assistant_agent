from pathlib import Path

import pytest
from llama_index.core import SimpleDirectoryReader

from assistant_agent.loaders.markdown import MarkdownReader


@pytest.mark.integration
def test_load_data_01():
    """SimpleDirectoryReader + MarkdownReader で docs/dataset_website/ を読み込めるか確認.

    観点1: 結果が 1 件以上であること
    観点2: 先頭 3 件の text および metadata が None でないこと
    """
    # 試験準備
    dataset_path = Path("docs/dataset_website/")
    if not dataset_path.exists():
        pytest.fail("docs/dataset_website/ が存在しないため失敗")

    # 試験実施
    reader = SimpleDirectoryReader(
        input_dir=str(dataset_path),
        file_extractor={".md": MarkdownReader()},
        recursive=True,
        required_exts=[".md"],
    )
    docs = reader.load_data()

    # 結果検証
    # 観点1
    assert len(docs) >= 1

    # 観点2
    for doc in docs[:3]:
        assert doc.text is not None
        assert doc.metadata is not None
