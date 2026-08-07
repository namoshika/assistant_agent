import asyncio
import itertools
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from assistant_agent.services.dispatcher import Dispatch, DispatcherChannel, DispatcherService
from assistant_agent.utils.absclass import AgentInvocation, Receiver


class TestDispatch:
    def test_str_01(self):
        """__str__() が予定を LLM 向けの英語1行テキストへ整形することを確認.

        観点1: dispatch_id・次回発火時刻・メッセージ内容を含むこと
        観点2: 100文字を超えるメッセージ内容は100文字に切り詰められること
        観点3: interval_seconds が ONE_SHOT の場合は "once"、それ以外は "every Ns" と
          表現され、内部表現（-1s 等）が現れないこと
        観点4: run_at は UTC ではなく JST（+09:00）に変換されて表示されること
        """
        # 試験準備
        long_content = "あ" * 150
        run_at = datetime(2026, 7, 30, 9, 0, 0, tzinfo=UTC)
        dispatch = Dispatch(
            dispatch_id="dispatch-1",
            agent_id="test-agent",
            prompt=long_content,
            interval_seconds=DispatcherService.ONE_SHOT,
            run_at=run_at,
        )

        # 試験実施
        text = str(dispatch)

        # 結果検証
        # 観点1
        assert "dispatch-1" in text
        # 観点2
        assert "あ" * 101 not in text
        assert "あ" * 100 in text
        # 観点3
        assert "once" in text
        # 観点4
        assert "2026-07-30T18:00:00+09:00" in text

        # 試験準備: 繰り返し予定
        dispatch.interval_seconds = 30

        # 試験実施
        text_repeat = str(dispatch)

        # 結果検証
        # 観点3
        assert "every 30s" in text_repeat
        assert "-1s" not in text_repeat


class TestDispatcherChannel:
    async def test_emit_01(self):
        """単発予定登録後、start() で emit() されることを確認.

        観点1: 単発予定（invoke_at() 等）を登録した DispatcherService を渡して start() すると、
          登録した prompt を content に持つ AgentInvocation が emit() されること
        観点2: emit される AgentInvocation の context["request_id"] が空でない文字列であること
        観点3: emit される AgentInvocation の context["agent_id"] が登録した Dispatch.agent_id と
          一致すること
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        service = AsyncMock(spec=DispatcherService)
        dispatch = Dispatch(
            dispatch_id="dispatch-1",
            agent_id="agent-a",
            prompt="test",
            interval_seconds=DispatcherService.ONE_SHOT,
            run_at=datetime.now(UTC),
        )
        service.pop_dispatch.side_effect = itertools.chain([[dispatch]], itertools.repeat([]))
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
        # 観点3
        assert emitted["context"]["agent_id"] == "agent-a"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    async def test_emit_02(self):
        """繰り返し予定が stop() まで複数回 emit() されることを確認.

        観点1: 一定間隔予定は stop() するまで繰り返し emit() されること
        観点2: 各回に emit() される AgentInvocation が別インスタンスであること
        観点3: 複数回 emit された AgentInvocation の context["request_id"] が毎回異なる値であること
        """
        # 試験準備
        received = MagicMock(spec=Receiver)
        service = AsyncMock(spec=DispatcherService)
        dispatch = Dispatch(
            dispatch_id="dispatch-1",
            agent_id="agent-a",
            prompt="test",
            interval_seconds=1,
            run_at=datetime.now(UTC),
        )
        service.pop_dispatch.return_value = [dispatch]
        channel = DispatcherChannel(service=service, poll_interval_seconds=0.02)
        channel.receiver = received

        # 試験実施
        channel.start()
        await asyncio.sleep(0.1)
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
        service = AsyncMock(spec=DispatcherService)
        dispatches = [
            Dispatch(
                dispatch_id=f"dispatch-{i}",
                agent_id="agent-a",
                prompt="test",
                interval_seconds=DispatcherService.ONE_SHOT,
                run_at=datetime.now(UTC),
            )
            for i in range(2)
        ]
        service.pop_dispatch.side_effect = itertools.chain([dispatches], itertools.repeat([]))
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
        service = AsyncMock(spec=DispatcherService)
        dispatch = Dispatch(
            dispatch_id="dispatch-1",
            agent_id="agent-a",
            prompt="test",
            interval_seconds=1,
            run_at=datetime.now(UTC),
        )
        service.pop_dispatch.return_value = [dispatch]
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
