from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from assistant_agent.entities import base
from assistant_agent.services.dispatcher import DispatcherService
from assistant_agent.store import PostgresStoreConnector
from assistant_agent.utils.dispatcher import DispatcherMaintenance


class TestDispatcherMaintenance:
    @pytest.mark.integration
    async def test_load_01(
        self,
        pg_conn: PostgresStoreConnector,
        pg_entity_dispatch: type[base.DispatchFields],
        tmp_path: Path,
    ):
        """DispatcherMaintenance.dump()・load() に対しテストすること.

        観点1: 指定 agent_id の予約だけがファイル集合で洗い替わり、他 agent_id の予約は残ること
        観点2: 不正ファイル・空・存在しないディレクトリで、DB の予約集合が正しく保たれること
        観点3: dump() の出力を load() すると、元の予約と一致すること
        """
        engine = pg_conn.get_engine()
        service = DispatcherService(dispatch_entity=pg_entity_dispatch, sa_engine=engine)
        maintenance = DispatcherMaintenance(
            dispatch_entity=pg_entity_dispatch, sa_engine=engine, root=tmp_path
        )
        now = datetime.now(UTC)

        # 試験準備: agent-a に単発・繰り返し予約、agent-b に別予約を登録
        once = await service.invoke_at("agent-a", "once prompt", now + timedelta(hours=1))
        recurring = await service.invoke_delay(
            "agent-a", "recurring prompt", delay_value=1, interval_value=3600
        )
        other_agent = await service.invoke_at("agent-b", "other", now + timedelta(hours=1))

        # 試験実施: agent-a を dump し、そのままの内容で load（観点3のラウンドトリップ）
        dumped_paths = await maintenance.dump("agent-a")
        loaded = await maintenance.load("agent-a")

        # 結果検証
        # 観点1・3
        assert {p.name for p in dumped_paths} == {
            f"{once.dispatch_id}.md",
            f"{recurring.dispatch_id}.md",
        }
        loaded_by_id = {d.dispatch_id: d for d in loaded}
        assert loaded_by_id[once.dispatch_id] == once
        assert loaded_by_id[recurring.dispatch_id] == recurring
        remaining_a = await service.list_dispatch("agent-a")
        assert {d.dispatch_id for d in remaining_a} == {once.dispatch_id, recurring.dispatch_id}
        remaining_b = await service.list_dispatch("agent-b")
        assert [d.dispatch_id for d in remaining_b] == [other_agent.dispatch_id]

        # 試験準備: agent-a のディレクトリへ不正ファイルを混入
        agent_dir = tmp_path / "agent-a"
        (agent_dir / "not-a-uuid.md").write_text(
            "---\nrun_at: '2026-08-14T09:00:00+00:00'\ninterval_seconds: -1\n---\nbad",
            encoding="utf-8",
        )

        # 試験実施・結果検証
        # 観点2: 不正ファイル（不正 YAML・必須キー不足・未知キー・naive run_at・不正
        # interval・dispatch_id 重複）混入時は例外が送出され、DB の予約集合は変わらない
        with pytest.raises(ValueError):
            await maintenance.load("agent-a")
        remaining_after_invalid = await service.list_dispatch("agent-a")
        assert {d.dispatch_id for d in remaining_after_invalid} == {
            once.dispatch_id,
            recurring.dispatch_id,
        }

        # 試験準備: 不正ファイルを除去し、空ディレクトリとして load
        (agent_dir / "not-a-uuid.md").unlink()
        for p in agent_dir.glob("*.md"):
            p.unlink()

        # 試験実施
        empty_loaded = await maintenance.load("agent-a")

        # 結果検証
        # 観点2: 空ディレクトリは agent_id の全削除として扱われる
        assert empty_loaded == []
        assert await service.list_dispatch("agent-a") == []

        # 試験実施・結果検証
        # 観点2: 存在しない agent_id ディレクトリはエラー
        with pytest.raises(ValueError):
            await maintenance.load("agent-nonexistent")
