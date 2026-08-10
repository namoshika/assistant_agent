import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Self
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from sqlalchemy import delete, func, literal_column, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from assistant_agent.entities import base
from assistant_agent.entities.postgres import DispatchEntity
from assistant_agent.store import PostgresStoreConnector
from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation
from assistant_agent.utils.context import ContextRegistry

_DISPATCHER_MESSAGE_TMPL = """\
# Dispatcher Channel
予約イベントが発火しました。

{{ dispatch }}
"""


class IntervalUnit(Enum):
    """時間の単位（LLM がそのままメンバー名を指定できるよう値も文字列にする）."""

    SECONDS = ("SECONDS", 1)
    MINUTES = ("MINUTES", 60)
    HOURS = ("HOURS", 3600)

    def __new__(cls, value: str, seconds: int) -> Self:
        """value（LLM 向けの文字列）と seconds（1単位あたりの秒数）を持つメンバーを生成する."""
        member = object.__new__(cls)
        member._value_ = value
        member.seconds = seconds
        return member

    seconds: int


@dataclass
class Dispatch:
    dispatch_id: str
    agent_id: str
    prompt: str
    interval_seconds: int
    run_at: datetime

    def __str__(self) -> str:
        """LLM への提示用に、予定の内容を整形する."""
        interval = (
            "once"
            if self.interval_seconds == DispatcherService.ONE_SHOT
            else f"every {self.interval_seconds}s"
        )
        run_at_jst = self.run_at.astimezone(ZoneInfo("Asia/Tokyo"))
        return (
            f"## Dispatch (dispatch_id: {self.dispatch_id})\n"
            f"### Metadata\n"
            f"- interval: {interval}\n"
            f"- run_at: {run_at_jst.isoformat()}\n\n"
            f"### Prompt\n"
            f"{self.prompt}"
        )


class DispatcherService:
    """予定（発信するプロンプトと発火タイミング）の登録・解除・参照を担うサービス.

    発火判定・AgentInvocation への変換・emit() は行わない（DispatcherChannel の責務）。
    """

    ONE_SHOT = -1

    def __init__(self, dispatch_entity: type, sa_engine: AsyncEngine) -> None:
        """予定を dispatch_entity のテーブル（sa_engine 経由）で管理する状態で初期化する."""
        self._entity = dispatch_entity
        self._engine = sa_engine

    async def setup(self) -> None:
        """dispatch_entity のテーブルが存在しなければ作成し、期限切れの単発予定を削除する.

        プロセス起動時に呼ばれる想定。停止中に run_at を過ぎた単発予定が emit されず
        一括発火しないよう、テーブル作成後に削除する（繰り返し予定は対象外）。
        """
        entity = self._entity
        async with self._engine.begin() as conn:
            await conn.run_sync(entity.metadata.create_all)
        async with AsyncSession(self._engine) as session:
            await session.execute(
                delete(entity).where(
                    entity.run_at <= datetime.now(UTC), entity.interval_seconds == self.ONE_SHOT
                )
            )
            await session.commit()

    async def invoke_at(self, agent_id: str, prompt: str, at: datetime) -> Dispatch:
        """指定した次回発火時刻（at）で単発予定を登録し、登録した Dispatch を返す."""
        return await self._register(agent_id, prompt, at, self.ONE_SHOT)

    async def invoke_delay(
        self,
        agent_id: str,
        prompt: str,
        delay_value: int = 10,
        delay_unit: IntervalUnit = IntervalUnit.SECONDS,
        interval_value: int = 0,
        interval_unit: IntervalUnit = IntervalUnit.MINUTES,
    ) -> Dispatch:
        """開始オフセット経過後に発火する予定を登録し、登録した Dispatch を返す.

        interval_value が 0 以下の場合は単発予定として登録する。既定値のまま呼ぶと
        「10秒後に単発発火」（即時発信相当）になる。
        """
        at = datetime.now(UTC) + timedelta(seconds=delay_value * delay_unit.seconds)
        interval_seconds = (
            self.ONE_SHOT if interval_value <= 0 else interval_value * interval_unit.seconds
        )
        return await self._register(agent_id, prompt, at, interval_seconds)

    async def cancel_dispatch(self, agent_id: str, dispatch_id: str) -> bool:
        """dispatch_id を指定して予定を解除する.

        存在しない dispatch_id を指定した場合は何もせず False を返す。
        """
        async with AsyncSession(self._engine) as session:
            deleted = (
                await session.scalars(
                    delete(self._entity)
                    .where(
                        self._entity.dispatch_id == dispatch_id, self._entity.agent_id == agent_id
                    )
                    .returning(self._entity)
                )
            ).all()
            found = len(deleted) > 0
            await session.commit()
            return found

    async def get_dispatch(self, agent_id: str, dispatch_id: str) -> Dispatch | None:
        """dispatch_id を指定して予定1件を取得する.

        存在しない dispatch_id を指定した場合は None を返す。
        """
        async with AsyncSession(self._engine) as session:
            row = await session.scalar(
                select(self._entity).where(
                    self._entity.dispatch_id == dispatch_id, self._entity.agent_id == agent_id
                )
            )
            return self._to_dispatch(row) if row is not None else None

    async def list_dispatch(self, agent_id: str) -> list[Dispatch]:
        """現在登録されている予定の一覧を返す."""
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.scalars(select(self._entity).where(self._entity.agent_id == agent_id))
            ).all()
            return [self._to_dispatch(row) for row in rows]

    async def pop_dispatch(self, now: datetime, agent_ids: list[str]) -> list[Dispatch]:
        """発火条件を満たす予定を取得し、後始末（単発削除・繰り返し次回時刻更新）まで行う.

        戻り値の Dispatch.run_at は単発・繰り返しいずれも「今回発火した時刻」を表す。
        agent_ids に含まれない agent_id の予定は対象外とし、DB に残す。
        """
        entity = self._entity
        due_condition = entity.run_at <= now
        agent_condition = entity.agent_id.in_(agent_ids)
        one_second = literal_column("interval '1 second'")
        elapsed_seconds = func.extract("epoch", now - entity.run_at)
        steps = func.floor(elapsed_seconds / entity.interval_seconds)
        advanced_seconds = steps * entity.interval_seconds + entity.interval_seconds
        new_run_at = entity.run_at + advanced_seconds * one_second

        async with AsyncSession(self._engine) as session:
            deleted = (
                await session.scalars(
                    delete(entity)
                    .where(due_condition, agent_condition, entity.interval_seconds == self.ONE_SHOT)
                    .returning(entity)
                )
            ).all()

            fired = (
                await session.scalars(
                    select(entity).where(
                        due_condition, agent_condition, entity.interval_seconds != self.ONE_SHOT
                    )
                )
            ).all()
            due = [self._to_dispatch(row) for row in (*deleted, *fired)]

            await session.execute(
                update(entity)
                .where(due_condition, agent_condition, entity.interval_seconds != self.ONE_SHOT)
                .values(run_at=new_run_at)
            )
            await session.commit()
            return due

    async def _register(
        self, agent_id: str, prompt: str, at: datetime, interval_seconds: int
    ) -> Dispatch:
        dispatch_id = str(uuid.uuid7())
        async with AsyncSession(self._engine) as session:
            session.add(
                self._entity(
                    dispatch_id=dispatch_id,
                    agent_id=agent_id,
                    prompt=prompt,
                    interval_seconds=interval_seconds,
                    run_at=at,
                )
            )
            await session.commit()
        return Dispatch(dispatch_id, agent_id, prompt, interval_seconds, at)

    @staticmethod
    def _to_dispatch(row: base.DispatchFields) -> Dispatch:
        return Dispatch(row.dispatch_id, row.agent_id, row.prompt, row.interval_seconds, row.run_at)


@ContextRegistry.register("dispatcher_service")
def build(store_conn: PostgresStoreConnector, **_: Any) -> DispatcherService:
    """DispatcherService を PostgresStoreConnector から生成する（PostgreSQL バックエンドで運用）."""
    return DispatcherService(dispatch_entity=DispatchEntity, sa_engine=store_conn.get_engine())


class DispatcherChannel(ActiveEmitter):
    """DispatcherService に登録された予定を、固定間隔でポーリングし emit() する実行エンジン."""

    def __init__(
        self, service: DispatcherService, poll_interval_seconds: float, agent_ids: list[str]
    ):
        """service（DispatcherService）とポーリング間隔（秒）、対象 agent_ids を構成する."""
        super().__init__()
        self._service = service
        self._poll_interval_seconds = poll_interval_seconds
        self._agent_ids = agent_ids
        self._task: asyncio.Task[None] | None = None
        self._logger = logging.getLogger(__name__)

    def start(self) -> None:
        """ポーリングを開始する（起動済みなら何もしない）."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        """ポーリングを停止する."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                for dispatch in await self._service.pop_dispatch(
                    datetime.now(UTC), self._agent_ids
                ):
                    try:
                        prompt = PromptTemplate.from_template(
                            _DISPATCHER_MESSAGE_TMPL, template_format="jinja2"
                        ).format(dispatch=dispatch)
                        invocation = AgentInvocation(
                            input={"messages": [HumanMessage(content=prompt)]},
                            context={
                                "request_id": str(uuid.uuid7()),
                                "agent_id": dispatch.agent_id,
                            },
                        )
                        self.emit(invocation)
                    except Exception:
                        self._logger.exception("Exception during dispatch emit")
            except Exception:
                self._logger.exception("Exception during dispatch polling")
            finally:
                await asyncio.sleep(self._poll_interval_seconds)
