from datetime import UTC, datetime, timedelta

import pytest

from assistant_agent.entities import base
from assistant_agent.services.dispatcher import Dispatch, DispatcherService, IntervalUnit
from assistant_agent.store import PostgresStoreConnector


class TestDispatcherService:
    @pytest.mark.integration
    async def test_invoke_at_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """invoke_at() で登録した予定が指定日時の前後で pop_dispatch() の対象になることを確認.

        観点1: 指定日時より前の pop_dispatch() には含まれず、指定日時以降の pop_dispatch() には
          含まれ、登録した Dispatch が返ること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        now = datetime.now(UTC)
        at = now + timedelta(seconds=10)

        # 試験実施
        dispatch = await service.invoke_at("agent-a", "test", at)

        # 結果検証
        # 観点1
        assert isinstance(dispatch, Dispatch)
        assert await service.pop_dispatch(now) == []
        due = await service.pop_dispatch(now + timedelta(seconds=11))
        assert len(due) == 1
        assert due[0].dispatch_id == dispatch.dispatch_id

    @pytest.mark.integration
    async def test_invoke_delay_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """invoke_delay() の開始オフセット・単位換算・単発判定・既定値を確認.

        観点1: 開始オフセット経過前の pop_dispatch() には含まれず、経過後の pop_dispatch() には
          含まれること
        観点2: IntervalUnit.MINUTES 等の単位指定が value * unit.seconds 秒として換算されること
        観点3: interval_value に 0 以下を指定すると単発予定になり、pop_dispatch() 後に削除される
          こと
        観点4: 引数を省略した場合、既定値により10秒後に単発発火する予定として登録されること
        """
        # 試験準備
        engine = pg_conn.get_engine()
        service = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)
        now = datetime.now(UTC)

        # 試験実施・結果検証
        # 観点1
        dispatch = await service.invoke_delay(
            "agent-a", "test", delay_value=5, delay_unit=IntervalUnit.SECONDS
        )
        assert await service.pop_dispatch(now + timedelta(seconds=1)) == []
        due = await service.pop_dispatch(now + timedelta(seconds=6))
        assert len(due) == 1
        assert due[0].dispatch_id == dispatch.dispatch_id

        # 観点2
        service2 = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)
        await service2.invoke_delay(
            "agent-a", "test", delay_value=1, delay_unit=IntervalUnit.MINUTES
        )
        assert await service2.pop_dispatch(now + timedelta(seconds=30)) == []
        assert len(await service2.pop_dispatch(now + timedelta(minutes=1, seconds=1))) == 1

        # 観点3
        service3 = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)
        await service3.invoke_delay("agent-a", "test", delay_value=1, interval_value=-1)
        await service3.invoke_delay("agent-a", "test", delay_value=1, interval_value=0)
        due3 = await service3.pop_dispatch(now + timedelta(seconds=2))
        assert len(due3) == 2
        assert await service3.pop_dispatch(now + timedelta(seconds=3)) == []

        # 観点4
        service4 = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)
        await service4.invoke_delay("agent-a", "test")
        assert await service4.pop_dispatch(now + timedelta(seconds=5)) == []
        assert len(await service4.pop_dispatch(now + timedelta(seconds=11))) == 1

    @pytest.mark.integration
    async def test_cancel_dispatch_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """cancel_dispatch() の解除を確認.

        観点1: cancel_dispatch した予定が以後の pop_dispatch に含まれず、戻り値が True になること。
          存在しない dispatch_id を指定しても例外が送出されず戻り値が False になること
        観点2: 他の agent_id が登録した dispatch_id を指定すると、存在しない場合と同様に
          False が返り、その予定は解除されずに残ること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        now = datetime.now(UTC)
        dispatch = await service.invoke_delay("agent-a", "test", delay_value=1)

        # 試験実施・結果検証
        # 観点1
        assert await service.cancel_dispatch("agent-a", "not-exist") is False

        # 試験実施・結果検証
        # 観点2
        assert await service.cancel_dispatch("agent-b", dispatch.dispatch_id) is False
        remaining_ids = {d.dispatch_id for d in await service.list_dispatch("agent-a")}
        assert dispatch.dispatch_id in remaining_ids

        # 試験実施・結果検証
        # 観点1
        assert await service.cancel_dispatch("agent-a", dispatch.dispatch_id) is True
        assert await service.pop_dispatch(now + timedelta(seconds=2)) == []

    @pytest.mark.integration
    async def test_get_dispatch_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """get_dispatch() による予定1件の取得を確認.

        観点1: 登録した予定が get_dispatch(agent_id, dispatch_id) で取得でき、内容が一致すること
        観点2: 存在しない dispatch_id を指定すると None が返り例外が送出されないこと
        観点3: 他の agent_id が登録した dispatch_id を指定すると、存在しない場合と区別できず
          None が返ること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        dispatch = await service.invoke_delay("agent-a", "test")

        # 試験実施・結果検証
        # 観点1
        found = await service.get_dispatch("agent-a", dispatch.dispatch_id)
        assert found == dispatch

        # 観点2
        assert await service.get_dispatch("agent-a", "not-exist") is None

        # 観点3
        assert await service.get_dispatch("agent-b", dispatch.dispatch_id) is None

    @pytest.mark.integration
    async def test_list_dispatch_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """list_dispatch() の取得を確認.

        観点1: 登録した予定が list_dispatch(agent_id) で取得でき、cancel_dispatch() 後は
          含まれないこと
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        dispatch = await service.invoke_delay("agent-a", "test")

        # 試験実施・結果検証
        # 観点1
        assert [d.dispatch_id for d in await service.list_dispatch("agent-a")] == [
            dispatch.dispatch_id
        ]
        await service.cancel_dispatch("agent-a", dispatch.dispatch_id)
        assert await service.list_dispatch("agent-a") == []

    @pytest.mark.integration
    async def test_list_dispatch_02(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """異なる agent_id で登録した予定同士が list_dispatch() で混ざらないことを確認.

        観点1: agent_id="agent-a" で登録した予定が list_dispatch("agent-b") の結果に現れない
          こと。逆方向も同様であること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        dispatch_a = await service.invoke_delay("agent-a", "test-a")
        dispatch_b = await service.invoke_delay("agent-b", "test-b")

        # 試験実施
        listed_a = await service.list_dispatch("agent-a")
        listed_b = await service.list_dispatch("agent-b")

        # 結果検証
        # 観点1
        assert [d.dispatch_id for d in listed_a] == [dispatch_a.dispatch_id]
        assert [d.dispatch_id for d in listed_b] == [dispatch_b.dispatch_id]

    @pytest.mark.integration
    async def test_pop_dispatch_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """pop_dispatch() の後始末（単発削除・繰り返し更新）を確認.

        観点1: 単発の予定は pop_dispatch() 後に取り除かれ、続けて呼んでも返らないこと
        観点2: 繰り返しの予定は残り、run_at が now より後へ更新されること
        観点3: 繰り返し予定の戻り値の run_at は、DB 更新後の次回時刻ではなく、
          今回発火した時刻であること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        now = datetime.now(UTC)
        await service.invoke_delay("agent-a", "test", delay_value=1, interval_value=-1)
        repeat_dispatch = await service.invoke_delay(
            "agent-a", "test", delay_value=1, interval_value=5, interval_unit=IntervalUnit.SECONDS
        )
        fire_at = now + timedelta(seconds=2)

        # 試験実施
        due = await service.pop_dispatch(fire_at)

        # 結果検証
        # 観点1
        assert len(due) == 2
        assert await service.pop_dispatch(fire_at) == []

        # 観点2
        remaining = await service.list_dispatch("agent-a")
        assert len(remaining) == 1
        assert remaining[0].run_at > fire_at

        # 観点3
        due_repeat = next(d for d in due if d.dispatch_id == repeat_dispatch.dispatch_id)
        assert due_repeat.run_at == repeat_dispatch.run_at
        assert due_repeat.run_at < remaining[0].run_at

    @pytest.mark.integration
    async def test_pop_dispatch_02(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """繰り返し予定の複数周期経過時の連続発火防止・累積ずれ防止を確認.

        観点1: 繰り返しの予定について複数周期ぶん経過した時刻で pop_dispatch() を呼んでも返るのは
          1件のみで、連続発火しないこと
        観点2: 更新後の run_at が「直前の発火予定時刻＋間隔」の系列上にあり、now 起点で
          ないこと
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        now = datetime.now(UTC)
        await service.invoke_delay(
            "agent-a", "test", delay_value=1, interval_value=2, interval_unit=IntervalUnit.SECONDS
        )
        first_fire_at = now + timedelta(seconds=1)

        # 試験実施
        due = await service.pop_dispatch(now + timedelta(seconds=10))

        # 結果検証
        # 観点1
        assert len(due) == 1

        # 観点2
        expected_next = first_fire_at + timedelta(seconds=2 * 5)
        remaining = await service.list_dispatch("agent-a")
        assert abs((remaining[0].run_at - expected_next).total_seconds()) < 1

    @pytest.mark.integration
    async def test_pop_dispatch_03(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """pop_dispatch() が発火することを確認.

        観点1: 異なる agent_id で登録した予定が pop_dispatch() 一発で両方とも取得されること
        観点2: 戻り値の各 Dispatch.agent_id が、それぞれ登録した agent_id と一致すること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        now = datetime.now(UTC)
        dispatch_a = await service.invoke_delay("agent-a", "test-a", delay_value=1)
        dispatch_b = await service.invoke_delay("agent-b", "test-b", delay_value=1)

        # 試験実施
        due = await service.pop_dispatch(now + timedelta(seconds=2))

        # 結果検証
        # 観点1
        assert {d.dispatch_id for d in due} == {dispatch_a.dispatch_id, dispatch_b.dispatch_id}
        # 観点2
        due_by_id = {d.dispatch_id: d for d in due}
        assert due_by_id[dispatch_a.dispatch_id].agent_id == "agent-a"
        assert due_by_id[dispatch_b.dispatch_id].agent_id == "agent-b"

    @pytest.mark.integration
    async def test_setup_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """setup() が期限切れの単発予定のみを削除することを確認.

        観点1: run_at が過去の単発予定は setup() 呼び出し後に list_dispatch() から消えること
        観点2: run_at が未来の単発予定は残ること
        観点3: run_at が過去の繰り返し予定は削除されず残ること
        """
        # 試験準備
        service = DispatcherService(
            dispatch_entity=pg_entity_dispatch, sa_engine=pg_conn.get_engine()
        )
        now = datetime.now(UTC)
        stale_dispatch = await service.invoke_at("agent-a", "stale", now - timedelta(seconds=10))
        future_dispatch = await service.invoke_at("agent-a", "future", now + timedelta(seconds=600))
        repeat_dispatch = await service.invoke_delay(
            "agent-a",
            "repeat",
            delay_value=-10,
            interval_value=5,
            interval_unit=IntervalUnit.SECONDS,
        )

        # 試験実施
        await service.setup()

        # 結果検証
        remaining_ids = {d.dispatch_id for d in await service.list_dispatch("agent-a")}
        # 観点1
        assert stale_dispatch.dispatch_id not in remaining_ids
        # 観点2
        assert future_dispatch.dispatch_id in remaining_ids
        # 観点3
        assert repeat_dispatch.dispatch_id in remaining_ids

    @pytest.mark.integration
    async def test_persistence_01(
        self, pg_conn: PostgresStoreConnector, pg_entity_dispatch: type[base.DispatchFields]
    ):
        """プロセス再起動を模した別インスタンスから、登録済みの予定を参照できることを確認（R2）.

        観点1: 同一の sa_engine を共有した別の DispatcherService インスタンスから、先のインスタンス
          で登録した未発火の予定が list_dispatch()・pop_dispatch() で正しく読み取れること
        """
        # 試験準備
        engine = pg_conn.get_engine()
        service = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)
        now = datetime.now(UTC)
        at_dispatch = await service.invoke_at("agent-a", "test-at", now + timedelta(seconds=10))
        delay_dispatch = await service.invoke_delay("agent-a", "test-delay", delay_value=1)

        # 試験実施
        restarted_service = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)

        # 結果検証
        # 観点1
        listed_ids = {d.dispatch_id for d in await restarted_service.list_dispatch("agent-a")}
        assert listed_ids == {at_dispatch.dispatch_id, delay_dispatch.dispatch_id}
        due = await restarted_service.pop_dispatch(now + timedelta(seconds=2))
        assert [d.dispatch_id for d in due] == [delay_dispatch.dispatch_id]
