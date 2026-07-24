import asyncio
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict, cast

import mlflow
from langgraph.checkpoint.memory import InMemorySaver

from assistant_agent.agents import build_agent
from assistant_agent.services.schedule import ScheduleService
from assistant_agent.utils import workflow

LOG_PATH = os.getenv("ASSISTANT_AGENT_LOG_PATH", "logs/solbot_history.log")
LOG_LEVEL = os.getenv("ASSISTANT_AGENT_LOG_LEVEL", "INFO")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")

# トレース用設定（agent_server.py と同様。logger.exception() から mlflow のトレースIDを
# 参照できるようにするため、常駐プロセスでも autolog を有効化する）
if MLFLOW_EXPERIMENT_ID is not None:
    mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
else:
    mlflow.set_experiment(experiment_name="agent-rag")
mlflow.bedrock.autolog()  # pyright: ignore[reportPrivateImportUsage]
mlflow.gemini.autolog()  # pyright: ignore[reportPrivateImportUsage]
mlflow.langchain.autolog(run_tracer_inline=True)  # pyright: ignore[reportPrivateImportUsage]


class AgentBotContext(TypedDict):
    """agent_bot.py が使うコンテキスト（CommonContext から取り出す項目の型）."""

    schedule_service: ScheduleService


@asynccontextmanager
async def _init_flow() -> AsyncGenerator[None]:
    lc_agent, ctx = build_agent(InMemorySaver())
    bot_ctx = cast(AgentBotContext, ctx)

    # フローを初期化
    schedule_service = bot_ctx["schedule_service"]
    agent = workflow.Agent(lc_agent, context=ctx)
    workflow.MergePipe([schedule_service.cron], agent)
    log_writer = workflow.LogWriter()
    workflow.BroadcastPipe(agent, [log_writer])

    # フローを起動・終了
    try:
        agent.start()
        schedule_service.cron.start()
        yield
    finally:
        agent.stop()
        schedule_service.cron.stop()


async def _amain() -> None:
    async with _init_flow():
        await asyncio.Event().wait()


def main() -> None:
    """agent_bot.py のエントリーポイント（pyproject.toml の project.scripts から呼ばれる）."""
    try:
        proj_dir = Path(__file__).resolve().parents[2]
        log_path = proj_dir / LOG_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=LOG_LEVEL,
            format="[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        )
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nStopping because Ctrl+C was received.")
        sys.exit(130)


if __name__ == "__main__":
    main()
