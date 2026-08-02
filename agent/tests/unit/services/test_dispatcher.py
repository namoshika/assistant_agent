import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from assistant_agent.services.dispatcher import (
    Dispatch,
    DispatcherChannel,
    DispatcherService,
    IntervalUnit,
)
from assistant_agent.utils.absclass import AgentInvocation, Receiver


class TestDispatch:
    def test_str_01(self):
        """__str__() が予定を LLM 向けの英語1行テキストへ整形することを確認.

        観点1: dispatch_id・次回発火時刻・メッセージ内容を含むこと
        観点2: 100文字を超えるメッセージ内容は100文字に切り詰められること
        観点3: interval_seconds が ONE_SHOT の場合は "once"、それ以外は "every Ns" と
          表現され、内部表現（-1s 等）が現れないこと
        """
        # 試験準備
        long_content = "あ" * 150
        next_fire_at = datetime(2026, 7, 30, 9, 0, 0, tzinfo=UTC)
        dispatch = Dispatch(
            dispatch_id="dispatch-1",
            prompt=long_content,
            interval_seconds=DispatcherService.ONE_SHOT,
            next_fire_at=next_fire_at,
        )

        # 試験実施
        text = str(dispatch)

        # 結果検証
        # 観点1
        assert "dispatch-1" in text
        assert next_fire_at.isoformat() in text
        # 観点2
        assert "あ" * 101 not in text
        assert "あ" * 100 in text
        # 観点3
        assert "once" in text

        # 試験準備: 繰り返し予定
        dispatch.interval_seconds = 30

        # 試験実施
        text_repeat = str(dispatch)

        # 結果検証
        # 観点3
        assert "every 30s" in text_repeat
        assert "-1s" not in text_repeat


class TestDispatcherService:
    def test_invoke_at_01(self):
        """invoke_at() で登録した予定が指定日時の前後で pop_dispatch() の対象になることを確認.

        観点1: 指定日時より前の pop_dispatch() には含まれず、指定日時以降の pop_dispatch() には
          含まれ、登録した Dispatch が返ること
        観点2: naive datetime を渡しても UTC として扱われ、例外なく比較できること
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)
        at_naive = (now + timedelta(seconds=10)).replace(tzinfo=None)

        # 試験実施
        dispatch = service.invoke_at("test", at_naive)

        # 結果検証
        # 観点1
        assert isinstance(dispatch, Dispatch)
        assert service.pop_dispatch(now) == []
        due = service.pop_dispatch(now + timedelta(seconds=11))
        assert len(due) == 1
        assert due[0].dispatch_id == dispatch.dispatch_id
        # 観点2
        assert due[0].next_fire_at.tzinfo is not None

    def test_invoke_at_02(self):
        """過去日時の許容誤差を確認.

        観点1: 現在時刻より1分以上前の日時を指定すると ValueError が送出され予定が登録されないこと
        観点2: 許容誤差内の日時（現在時刻に近い過去）は正常に登録できること
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)

        # 試験実施・結果検証
        # 観点1
        try:
            service.invoke_at("test", now - timedelta(minutes=5))
            raised = False
        except ValueError:
            raised = True
        assert raised
        assert service.list_dispatch() == []

        # 観点2
        dispatch = service.invoke_at("test", now - timedelta(seconds=1))
        assert dispatch.dispatch_id in {d.dispatch_id for d in service.list_dispatch()}

    def test_invoke_delay_01(self):
        """invoke_delay() の開始オフセット・単位換算・既定値を確認.

        観点1: 開始オフセット経過前の pop_dispatch() には含まれず、経過後の pop_dispatch() には
          含まれること
        観点2: IntervalUnit.MINUTES 等の単位指定が value * unit.value 秒として換算されること
        観点3: 引数を省略した場合、既定値により10秒後に単発発火する予定として登録されること
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)

        # 試験実施
        dispatch = service.invoke_delay("test", delay_value=5, delay_unit=IntervalUnit.SECONDS)

        # 結果検証
        # 観点1
        assert service.pop_dispatch(now + timedelta(seconds=1)) == []
        due = service.pop_dispatch(now + timedelta(seconds=6))
        assert len(due) == 1
        assert due[0].dispatch_id == dispatch.dispatch_id

        # 観点2
        service2 = DispatcherService()
        service2.invoke_delay("test", delay_value=1, delay_unit=IntervalUnit.MINUTES)
        assert service2.pop_dispatch(now + timedelta(seconds=30)) == []
        assert len(service2.pop_dispatch(now + timedelta(minutes=1, seconds=1))) == 1

        # 観点3
        service3 = DispatcherService()
        service3.invoke_delay("test")
        assert service3.pop_dispatch(now + timedelta(seconds=5)) == []
        assert len(service3.pop_dispatch(now + timedelta(seconds=11))) == 1

    def test_invoke_delay_02(self):
        """interval_value に 0 以下を指定した場合の単発判定を確認.

        観点1: interval_value に 0 以下（-1・0）を指定した場合、開始オフセット経過後に1度だけ
          発火し、以後 pop_dispatch() に含まれないこと
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)
        service.invoke_delay("test", delay_value=1, interval_value=-1)
        service.invoke_delay("test", delay_value=1, interval_value=0)

        # 試験実施
        due = service.pop_dispatch(now + timedelta(seconds=2))

        # 結果検証
        # 観点1
        assert len(due) == 2
        assert service.pop_dispatch(now + timedelta(seconds=3)) == []

    def test_cancel_dispatch_01(self):
        """cancel_dispatch() の解除を確認.

        観点1: cancel_dispatch(dispatch_id) した予定が以後の pop_dispatch() に含まれず、戻り値が
          True になること。存在しない dispatch_id を指定しても例外が送出されず戻り値が False に
          なること
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)
        dispatch = service.invoke_delay("test", delay_value=1)

        # 試験実施・結果検証
        # 観点1
        assert service.cancel_dispatch("not-exist") is False
        assert service.cancel_dispatch(dispatch.dispatch_id) is True
        assert service.pop_dispatch(now + timedelta(seconds=2)) == []

    def test_list_dispatch_01(self):
        """list_dispatch() の取得を確認.

        観点1: 登録した予定が list_dispatch() で取得でき、cancel_dispatch() 後は含まれないこと
        """
        # 試験準備
        service = DispatcherService()
        dispatch = service.invoke_delay("test")

        # 試験実施・結果検証
        # 観点1
        assert [d.dispatch_id for d in service.list_dispatch()] == [dispatch.dispatch_id]
        service.cancel_dispatch(dispatch.dispatch_id)
        assert service.list_dispatch() == []

    def test_pop_dispatch_01(self):
        """pop_dispatch() の後始末（単発削除・繰り返し更新）を確認.

        観点1: 単発の予定は pop_dispatch() 後に取り除かれ、続けて呼んでも返らないこと
        観点2: 繰り返しの予定は残り、next_fire_at が now より後へ更新されること
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)
        service.invoke_delay("test", delay_value=1, interval_value=-1)
        service.invoke_delay(
            "test",
            delay_value=1,
            interval_value=5,
            interval_unit=IntervalUnit.SECONDS,
        )
        fire_at = now + timedelta(seconds=2)

        # 試験実施
        due = service.pop_dispatch(fire_at)

        # 結果検証
        # 観点1
        assert len(due) == 2
        assert service.pop_dispatch(fire_at) == []

        # 観点2
        remaining = service.list_dispatch()
        assert len(remaining) == 1
        assert remaining[0].next_fire_at > fire_at

    def test_pop_dispatch_02(self):
        """繰り返し予定の複数周期経過時の連続発火防止・累積ずれ防止を確認.

        観点1: 繰り返しの予定について複数周期ぶん経過した時刻で pop_dispatch() を呼んでも返るのは
          1件のみで、連続発火しないこと
        観点2: 更新後の next_fire_at が「直前の発火予定時刻＋間隔」の系列上にあり、now 起点で
          ないこと
        """
        # 試験準備
        service = DispatcherService()
        now = datetime.now(UTC)
        service.invoke_delay(
            "test",
            delay_value=1,
            interval_value=2,
            interval_unit=IntervalUnit.SECONDS,
        )
        first_fire_at = now + timedelta(seconds=1)

        # 試験実施
        due = service.pop_dispatch(now + timedelta(seconds=10))

        # 結果検証
        # 観点1
        assert len(due) == 1

        # 観点2
        expected_next = first_fire_at + timedelta(seconds=2 * 5)
        remaining = service.list_dispatch()
        assert abs((remaining[0].next_fire_at - expected_next).total_seconds()) < 1


class TestDispatcherChannel:
    async def test_emit_01(self):
        """単発予定登録後、start() で emit() されることを確認.

        観点1: 単発予定（invoke_at() 等）を登録した DispatcherService を渡して start() すると、
          登録した prompt を content に持つ AgentInvocation が emit() されること
        観点2: emit される AgentInvocation の context["request_id"] が空でない文字列であること
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        service = DispatcherService()
        service.invoke_delay("test", delay_value=0)
        channel = DispatcherChannel(service=service, poll_interval_seconds=0.01)
        channel.receiver = received

        # 試験実施
        channel.start()
        await asyncio.sleep(0.05)
        channel.stop()

        # 結果検証
        # 観点1
        received.on_received.assert_called_once()
        emitted: AgentInvocation = received.on_received.call_args[0][0]
        assert "test" in emitted["input"]["messages"][-1].content
        # 観点2
        assert isinstance(emitted["context"]["request_id"], str)  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert emitted["context"]["request_id"] != ""  # pyright: ignore[reportTypedDictNotRequiredAccess]

    async def test_emit_02(self):
        """繰り返し予定が stop() まで複数回 emit() されることを確認.

        観点1: 一定間隔予定は stop() するまで繰り返し emit() されること
        観点2: 各回に emit() される AgentInvocation が別インスタンスであること
        観点3: 複数回 emit された AgentInvocation の context["request_id"] が毎回異なる値であること
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        service = DispatcherService()
        service.invoke_delay(
            "test",
            delay_value=0,
            interval_value=1,
            interval_unit=IntervalUnit.SECONDS,
        )
        channel = DispatcherChannel(service=service, poll_interval_seconds=0.02)
        channel.receiver = received

        # 試験実施
        channel.start()
        await asyncio.sleep(2.1)
        channel.stop()

        # 結果検証
        # 観点1
        assert received.on_received.call_count >= 2
        # 観点2
        emitted_1 = received.on_received.call_args_list[0][0][0]
        emitted_2 = received.on_received.call_args_list[1][0][0]
        assert emitted_1 is not emitted_2
        # 観点3
        assert emitted_1["context"]["request_id"] != emitted_2["context"]["request_id"]

    async def test_emit_03(self):
        """配信先の例外発生時もポーリングが継続することを確認.

        観点1: 配信先の on_received() が例外を送出してもポーリングループが継続し後続の予定が
          発火すること
        観点2: 例外を送出した単発予定が再発火しないこと
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        received.on_received.side_effect = [RuntimeError("boom"), None]
        service = DispatcherService()
        service.invoke_delay("test", delay_value=0)
        service.invoke_delay("test", delay_value=0)
        channel = DispatcherChannel(service=service, poll_interval_seconds=0.01)
        channel.receiver = received

        # 試験実施
        channel.start()
        await asyncio.sleep(0.05)
        channel.stop()

        # 結果検証
        # 観点1
        assert received.on_received.call_count == 2
        # 観点2
        await asyncio.sleep(0.05)
        assert received.on_received.call_count == 2

    async def test_start_01(self):
        """stop() 後は新規の emit() が発生しないことを確認.

        観点1: stop() を呼んだ後は emit() の呼び出し回数が増えないこと
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        service = DispatcherService()
        service.invoke_delay(
            "test",
            delay_value=0,
            interval_value=1,
            interval_unit=IntervalUnit.SECONDS,
        )
        channel = DispatcherChannel(service=service, poll_interval_seconds=0.01)
        channel.receiver = received

        # 試験実施
        channel.start()
        await asyncio.sleep(0.03)
        channel.stop()
        count_after_stop = received.on_received.call_count
        await asyncio.sleep(0.05)

        # 結果検証
        # 観点1
        assert received.on_received.call_count == count_after_stop
