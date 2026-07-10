from pathlib import Path

import pytest
from langchain_community.document_loaders import DirectoryLoader

from assistant_agent.loaders.markdown import MarkdownLoader


@pytest.mark.integration
def test_load_data_01():
    """DirectoryLoader + MarkdownLoader で docs/dataset_website/ を読み込めるか確認.

    観点1: 結果が 1 件以上であること
    観点2: 先頭 3 件の page_content および metadata が None でないこと
    """
    # 試験準備
    dataset_path = Path("docs/dataset_website/")
    if not dataset_path.exists():
        pytest.fail("docs/dataset_website/ が存在しないため失敗")

    # 試験実施
    reader = DirectoryLoader(
        str(dataset_path),
        glob="**/*.md",
        loader_cls=MarkdownLoader,  # type: ignore[arg-type]
    )
    docs = reader.load()

    # 結果検証
    # 観点1
    assert len(docs) >= 1

    # 観点2
    for doc in docs[:3]:
        assert doc.page_content is not None
        assert doc.metadata is not None
