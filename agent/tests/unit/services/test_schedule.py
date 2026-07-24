import asyncio
from unittest.mock import MagicMock

from assistant_agent.services.schedule import CronChannel, ScheduleService
from assistant_agent.utils.absclass import Receiver


class TestCronChannel:
    async def test_emit_01(self):
        """一定間隔で emit() が呼ばれることを確認.

        観点1: interval_seconds を短く指定して start() すると、指定間隔で emit() が呼ばれる
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        cron = CronChannel(interval_seconds=0.01)
        cron.receiver = received

        # 試験実施
        cron.start()
        await asyncio.sleep(0.05)
        cron.stop()

        # 結果検証
        # 観点1
        assert received.on_received.call_count >= 1

    async def test_start_01(self):
        """stop() 後は新規の emit() が発生しないことを確認.

        観点1: stop() を呼んだ後は emit() の呼び出し回数が増えないこと
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        cron = CronChannel(interval_seconds=0.01)
        cron.receiver = received

        # 試験実施
        cron.start()
        await asyncio.sleep(0.05)
        cron.stop()
        count_after_stop = received.on_received.call_count
        await asyncio.sleep(0.05)

        # 結果検証
        # 観点1
        assert received.on_received.call_count == count_after_stop


class TestScheduleService:
    def test_construct_01(self):
        """ScheduleService が CronChannel を保持することを確認.

        観点1: 構築した ScheduleService.cron が CronChannel のインスタンスであること
        """
        # 試験準備
        service = ScheduleService(cron=CronChannel())

        # 結果検証
        # 観点1
        assert isinstance(service.cron, CronChannel)
