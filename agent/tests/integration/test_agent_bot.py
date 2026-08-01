import asyncio
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import HumanMessage
from pytest_mock import MockerFixture

from assistant_agent import agent_bot
from assistant_agent.services.dispatcher import Dispatch, DispatcherService


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
    """Agent が起動されることを確認.

    観点1: Agent が invoke（応答をログ出力）されること
    """
    # 試験準備
    dispatch = Dispatch(
        dispatch_id="test-dispatch",
        invocation={"input": {"messages": [HumanMessage(content="面白い話をして。")]}},
        interval_seconds=-1,
        next_fire_at=datetime.now(ZoneInfo("Asia/Tokyo")),
    )
    service = mocker.Mock(spec=DispatcherService)
    service.pop_dispatch.side_effect = [[dispatch], []]
    mocker.patch("assistant_agent.services.dispatcher.DispatcherService", return_value=service)

    # 試験実施
    try:
        await asyncio.wait_for(agent_bot._amain(), timeout=20)
    except TimeoutError:
        pass

    # 結果検証
    # 観点1
    service.pop_dispatch.assert_called()
    assert log_path.exists()
    assert log_path.read_text().strip()
