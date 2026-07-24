import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from assistant_agent import agent_bot


@pytest.fixture
def log_path(tmp_path: Path) -> Iterator[Path]:
    """LogWriter（agents/workflow.py の logger）へ一時ファイル宛の FileHandler を設定する.

    テスト終了後、追加したハンドラを取り除く。
    """
    path = tmp_path / "solbot_history.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger = logging.getLogger("assistant_agent.utils.workflow")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield path
    logger.removeHandler(handler)
    handler.close()


@pytest.mark.integration
async def test_amain_01(mocker: MockerFixture, log_path: Path):
    """Solbot 起動時、CronChannel の発火をトリガーに Agent が応答しログ出力することを確認.

    観点1: _amain() を実行すると、LogWriter を通じて Agent の応答が
        ログファイルへ書き出されること
    """
    # 試験実施
    try:
        await asyncio.wait_for(agent_bot._amain(), timeout=30)
    except TimeoutError:
        pass

    # 結果検証
    # 観点1
    assert log_path.exists()
    res = log_path.read_text().strip()
    assert res
