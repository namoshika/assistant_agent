import asyncio
import logging
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pytest_mock import MockerFixture

from assistant_agent import agent_bot
from assistant_agent.services.dispatcher import Dispatch, DispatcherService
from assistant_agent.store import PostgresStoreConnector


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

    観点1: agent_ev が invoke（Dispatcher 応答をログ出力）されること
    観点2: init_harness() が返す sync_request_channel へ emit_and_wait() でメッセージを送ると、
        agent_ch 経由で実グラフの応答が返ること
    """
    # 試験準備
    dispatch = Dispatch(
        dispatch_id="test-dispatch",
        agent_id="test-agent",
        prompt="面白い話をして。",
        interval_seconds=-1,
        run_at=datetime.now(ZoneInfo("Asia/Tokyo")),
    )
    service = mocker.AsyncMock(spec=DispatcherService)
    service.pop_dispatch.side_effect = [[dispatch], []]
    mocker.patch("assistant_agent.services.dispatcher.DispatcherService", return_value=service)

    # 試験実施
    async with agent_bot.init_harness("sample", "test-agent") as sync_request_channel:
        await asyncio.sleep(65)
        msg_out = await sync_request_channel.emit_and_wait(
            [HumanMessage(content="こんにちは。自己紹介してください。")], timeout_seconds=60
        )

    # 結果検証
    # 観点1
    service.pop_dispatch.assert_called()
    assert log_path.exists()
    assert log_path.read_text().strip()
    # 観点2
    assert msg_out.content


@pytest.mark.integration
async def test_find_latest_thread_id_01(pg_conn: PostgresStoreConnector) -> None:
    """find_latest_thread_id() が agent_id の prefix で絞り込み、最新の thread_id を返すことを確認.

    観点1: 同一 agent_id の複数 thread_id のうち、最も新しく生成されたものを返すこと
    観点2: 別の agent_id に属する thread_id は結果に混ざらないこと
    観点3: 該当する thread_id が無い agent_id では None を返すこと
    """
    # 試験準備
    agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"
    other_agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"
    thread_id_old = f"{agent_id}:{uuid.uuid7()}"
    thread_id_new = f"{agent_id}:{uuid.uuid7()}"
    thread_id_other = f"{other_agent_id}:{uuid.uuid7()}"

    async with pg_conn.get_psycopg_pool() as pool:
        saver = AsyncPostgresSaver(conn=pool)
        await saver.setup()
        for tid in (thread_id_old, thread_id_new, thread_id_other):
            checkpoint: Checkpoint = empty_checkpoint()
            config: RunnableConfig = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
            await saver.aput(config, checkpoint, {}, {})

    # 試験実施、結果検証
    # 観点1～2
    result = await agent_bot.find_latest_thread_id(pg_conn.get_engine(), agent_id)
    assert result == thread_id_new

    # 観点3
    result_none = await agent_bot.find_latest_thread_id(pg_conn.get_engine(), "no-such-agent")
    assert result_none is None
