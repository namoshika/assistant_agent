import asyncio
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict, cast

import mlflow
import typer
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import TTLConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

import assistant_agent.services  # noqa: F401  登録発火（ContextRegistry へ factory を登録）
from assistant_agent import agents
from assistant_agent.entities.checkpoint import CheckpointEntity
from assistant_agent.services.discord import DiscordChannel, DiscordService
from assistant_agent.services.dispatcher import DispatcherChannel, DispatcherService
from assistant_agent.store import PostgresStoreConnector
from assistant_agent.utils import context, workflow

LOG_DIR = os.getenv("AA_LOG_PATH", "logs")
LOG_LEVEL = os.getenv("AA_LOG_LEVEL", "INFO")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")
STORE_TTL_DEFAULT_DAYS = 30  # 仮想ファイルシステム上のアイテムの既定 TTL（日数）
STORE_TTL_SWEEP_INTERVAL_MINUTES = 60  # TTL 期限切れアイテムを自動削除する間隔（分）


class AgentBotContext(TypedDict):
    """agent_bot.py が使うコンテキスト（CommonContext から取り出す項目の型）."""

    discord_service: DiscordService
    dispatcher_service: DispatcherService


async def find_latest_thread_id(engine: AsyncEngine, agent_id: str) -> str | None:
    """Checkpoints から agent_id に属する thread_id を降順（uuid7 は時刻順）で1件取得する.

    プロセス再起動時に直前の thread_id から再開させるための補助。
    thread_id が f"{agent_id}:..." の形式で払い出されることが前提。
    """
    async with engine.connect() as conn:
        stmt = (
            select(CheckpointEntity.thread_id)
            .where(CheckpointEntity.thread_id.startswith(f"{agent_id}:"))
            .order_by(CheckpointEntity.thread_id.desc())
            .limit(1)
        )
        return (await conn.execute(stmt)).scalar()


@asynccontextmanager
async def init_harness(
    module_name: str, agent_id: str | None
) -> AsyncGenerator[workflow.SyncRequestChannel]:
    """エージェントハーネスを初期化."""
    agent_id = agent_id or module_name
    # ログ出力を構成
    proj_dir = Path(__file__).resolve().parents[2]
    log_path = proj_dir / LOG_DIR / f"solbot_{agent_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format=f"%(asctime)s [%(process)d] [%(levelname)s] [%(filename)s] [{agent_id}]: %(message)s",  # noqa: E501
        datefmt="%Y-%m-%d %H:%M:%S%z",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    # トレース用設定（agent_server.py と同様。logger.exception() から mlflow のトレースIDを
    # 参照できるようにするため、常駐プロセスでも autolog を有効化する）
    if MLFLOW_EXPERIMENT_ID is not None:
        mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
    else:
        mlflow.set_experiment(experiment_name="agent-rag")
    # mlflow.openai.autolog()  # pyright: ignore[reportPrivateImportUsage]
    mlflow.gemini.autolog()  # pyright: ignore[reportPrivateImportUsage]
    mlflow.langchain.autolog(run_tracer_inline=True)  # pyright: ignore[reportPrivateImportUsage]

    # コンテキスト構築
    store_conn = PostgresStoreConnector(os.environ["AA_PG_CONNECTION_STRING"])
    ctx = context.ContextRegistry.build(store_conn=store_conn)
    bot_ctx = cast(AgentBotContext, ctx)

    async with store_conn.get_psycopg_pool() as pool:
        checkpointer = AsyncPostgresSaver(conn=pool)
        pg_store = AsyncPostgresStore(
            conn=pool,
            ttl=TTLConfig(
                default_ttl=STORE_TTL_DEFAULT_DAYS * 24 * 60,  # 分単位
                refresh_on_read=True,
                sweep_interval_minutes=STORE_TTL_SWEEP_INTERVAL_MINUTES,
            ),
        )
        await checkpointer.setup()
        await pg_store.setup()
        await bot_ctx["dispatcher_service"].setup()
        await pg_store.start_ttl_sweeper()

        # エージェントを構築
        # NOTE: langchain-openai の ChatOpenAI._resolve_model_profile() は model_name を
        # 内部テーブルへの完全一致で引くため、Bedrock Mantle 用の "openai." 接頭辞付き
        # モデルID（例: "openai.gpt-5.6-luna"）では profile が解決できず空になる。
        # 接頭辞なしのモデル名で profile だけを解決し、明示的に渡すことで回避する。
        prf = init_chat_model("openai:gpt-5.6-luna", use_responses_api=True).profile
        llm = init_chat_model("openai:openai.gpt-5.6-luna", use_responses_api=True, profile=prf)
        latest_thread_id = await find_latest_thread_id(store_conn.get_engine(), agent_id)
        agent = agents.get_agent(
            module_name, agent_id, latest_thread_id, llm, ctx, checkpointer, pg_store
        )

        # フローを初期化
        sync_request_channel = workflow.SyncRequestChannel()
        dispatcher_channel = DispatcherChannel(
            service=bot_ctx["dispatcher_service"],
            poll_interval_seconds=60.0,
            agent_ids=[agent_id],
        )
        discord_channel = DiscordChannel(service=bot_ctx["discord_service"])
        log_writer = workflow.LogWriter()
        workflow.MergePipe([sync_request_channel, dispatcher_channel, discord_channel], agent)
        workflow.BroadcastPipe(agent, [sync_request_channel, log_writer])

        # フローを起動・終了
        logger.info(
            f"Start Assistant Agent Loop (agent_id: {agent_id}, thread_id: {latest_thread_id})"
        )
        try:
            agent.start()
            sync_request_channel.start()
            dispatcher_channel.start()
            discord_channel.start()
            yield sync_request_channel
        finally:
            agent.stop()
            sync_request_channel.stop()
            dispatcher_channel.stop()
            discord_channel.stop()
            await pg_store.stop_ttl_sweeper()


async def _amain(module_name: str = "sample", agent_id: str | None = None) -> None:
    async with init_harness(module_name, agent_id):
        await asyncio.Event().wait()


app = typer.Typer()


def _validate_agent_module(value: str) -> str:
    if not agents.module_exists(value):
        raise typer.BadParameter(f"No agent module named {value!r} in agents/")
    return value


@app.command()
def main(
    agent: str = typer.Option(
        "sample",
        "--agent",
        callback=_validate_agent_module,
        help="起動するエージェントのモジュール名",
    ),
    overwrite_agent_id: str | None = typer.Option(
        None,
        "--overwrite_agent_id",
        help="thread_id・backend namespace 等に使う agent_id（省略時は --agent の値）",
    ),
) -> None:
    """Solbot コマンドの処理本体（pyproject.toml の project.scripts が指す app 経由で呼ばれる）."""
    try:
        asyncio.run(_amain(agent, overwrite_agent_id))
    except KeyboardInterrupt:
        print("\nStopping because Ctrl+C was received.")
        sys.exit(130)


if __name__ == "__main__":
    app()
