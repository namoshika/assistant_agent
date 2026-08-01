import asyncio
import copy
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Self

from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation
from assistant_agent.utils.context import ContextRegistry


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
    invocation: AgentInvocation
    interval_seconds: int
    next_fire_at: datetime

    def __str__(self) -> str:
        """LLM への提示用に、予定の内容を英語の1行テキストへ整形する（LLM の精度向上のため）."""
        content = str(self.invocation["input"]["messages"][-1].content)
        content = content[:100]
        interval = (
            "once"
            if self.interval_seconds == DispatcherService.ONE_SHOT
            else f"every {self.interval_seconds}s"
        )
        return (
            f"dispatch_id={self.dispatch_id}, interval={interval}, "
            f"next_fire_at={self.next_fire_at.isoformat()}, content={content}"
        )


class DispatcherService:
    """予定（発信する AgentInvocation と発火タイミング）の登録・解除・参照を担うサービス.

    発火判定・emit() は行わない（DispatcherChannel の責務）。
    """

    ONE_SHOT = -1
    _PAST_TOLERANCE = timedelta(minutes=1)

    def __init__(self) -> None:
        """予定を保持しない空の状態で初期化する."""
        self._dispatches: dict[str, Dispatch] = {}

    def invoke_at(self, invocation: AgentInvocation, at: datetime) -> Dispatch:
        """指定日時に発火する単発予定を登録し、登録した Dispatch を返す.

        naive datetime は UTC とみなして正規化する。現在時刻より _PAST_TOLERANCE を超えて
        過去の日時は ValueError を送出する。
        """
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        if at < now - self._PAST_TOLERANCE:
            raise ValueError(f"過去の日時は指定できません: {at.isoformat()}")
        return self._register(invocation, at, self.ONE_SHOT)

    def invoke_delay(
        self,
        invocation: AgentInvocation,
        delay_value: int = 10,
        delay_unit: IntervalUnit = IntervalUnit.SECONDS,
        interval_value: int = 0,
        interval_unit: IntervalUnit = IntervalUnit.MINUTES,
    ) -> Dispatch:
        """開始オフセット経過後に発火する予定を登録し、登録した Dispatch を返す.

        interval_value が 0 以下の場合は単発予定として登録する。既定値のまま呼ぶと
        「10秒後に単発発火」（即時発信相当）になる。
        """
        now = datetime.now(UTC)
        at = now + timedelta(seconds=delay_value * delay_unit.seconds)
        interval_seconds = (
            self.ONE_SHOT if interval_value <= 0 else interval_value * interval_unit.seconds
        )
        return self._register(invocation, at, interval_seconds)

    def cancel_dispatch(self, dispatch_id: str) -> bool:
        """dispatch_id を指定して予定を解除する.

        存在しない dispatch_id を指定した場合は何もせず False を返す。
        """
        if dispatch_id not in self._dispatches:
            return False
        del self._dispatches[dispatch_id]
        return True

    def list_dispatch(self) -> list[Dispatch]:
        """現在保持している予定の一覧を返す."""
        return list(self._dispatches.values())

    def pop_dispatch(self, now: datetime) -> list[Dispatch]:
        """発火条件を満たす予定を取得し、後始末（単発削除・繰り返し次回時刻更新）まで行う.

        繰り返しの次回時刻は、直前の発火予定時刻を起点に間隔を加算し、now を超えるまで
        繰り返して更新する（ポーリング周期の累積ずれと連続発火を避けるため）。
        """
        due: list[Dispatch] = []
        for dispatch_id in list(self._dispatches):
            dispatch = self._dispatches[dispatch_id]
            if dispatch.next_fire_at > now:
                continue
            due.append(dispatch)
            if dispatch.interval_seconds == self.ONE_SHOT:
                del self._dispatches[dispatch_id]
            else:
                while dispatch.next_fire_at <= now:
                    dispatch.next_fire_at += timedelta(seconds=dispatch.interval_seconds)
        return due

    def _register(
        self, invocation: AgentInvocation, at: datetime, interval_seconds: int
    ) -> Dispatch:
        dispatch_id = str(uuid.uuid7())
        dispatch = Dispatch(
            dispatch_id=dispatch_id,
            invocation=invocation,
            interval_seconds=interval_seconds,
            next_fire_at=at,
        )
        self._dispatches[dispatch_id] = dispatch
        return dispatch


@ContextRegistry.register("dispatcher_service")
def build(**_: Any) -> DispatcherService:
    """DispatcherService を生成する（他コンテキストに依存しない）."""
    return DispatcherService()


class DispatcherChannel(ActiveEmitter):
    """DispatcherService に登録された予定を、固定間隔でポーリングし emit() する実行エンジン."""

    def __init__(self, service: DispatcherService, poll_interval_seconds: float = 1.0):
        """service（DispatcherService）とポーリング間隔（秒）を構成する."""
        super().__init__()
        self._service = service
        self._poll_interval_seconds = poll_interval_seconds
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
                for dispatch in self._service.pop_dispatch(datetime.now(UTC)):
                    try:
                        self.emit(copy.deepcopy(dispatch.invocation))
                    except Exception:
                        self._logger.exception("Exception during dispatch emit")
            except Exception:
                self._logger.exception("Exception during dispatch polling")
            finally:
                await asyncio.sleep(self._poll_interval_seconds)
