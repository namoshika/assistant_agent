import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

from assistant_agent.utils.absclass import ActiveEmitter
from assistant_agent.utils.context import ContextRegistry

prompt_cron = """
面白い話をして。
現在時刻: {now}
"""


class CronChannel(ActiveEmitter):
    """一定間隔でトリガーメッセージを発信する ActiveEmitter."""

    def __init__(self, interval_seconds: float = 120):
        """interval_seconds（発信間隔・秒）・message（発信する固定メッセージ）を構成する."""
        super().__init__()
        self._interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """定期発信を開始する（起動済みなら何もしない）."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        """定期発信を停止する."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while True:
            dt_now = datetime.now(ZoneInfo("Asia/Tokyo"))
            self.emit(HumanMessage(content=prompt_cron.format(now=dt_now.isoformat())))
            await asyncio.sleep(self._interval_seconds)


class ScheduleService:
    """CronChannel を保持するサービス."""

    def __init__(self, cron: CronChannel):
        """CronChannel を構成する."""
        self.cron = cron


@ContextRegistry.register("schedule_service")
def build(**_: Any) -> ScheduleService:
    """ScheduleService を生成する."""
    return ScheduleService(cron=CronChannel())
