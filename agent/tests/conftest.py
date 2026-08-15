import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def profile_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """AA_PROFILE_DIR をテスト用一時ディレクトリへ差し替える.

    本番・デバッグ実行用のプロファイルディレクトリを汚さないようにする。
    """
    profile_dir = tmp_path_factory.mktemp("assistant_agent_profile")
    os.environ["AA_PROFILE_DIR"] = str(profile_dir)
    yield profile_dir
    del os.environ["AA_PROFILE_DIR"]
