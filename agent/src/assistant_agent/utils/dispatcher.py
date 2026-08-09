import shutil
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from assistant_agent.loaders.markdown import FRONT_MATTER_DELIMITER, MarkdownLoader
from assistant_agent.services.dispatcher import Dispatch, DispatcherService


class _DispatchFrontMatter(BaseModel):
    """予約ファイルのフロントマター schema（agent_id・dispatch_id はパス側で保持する）."""

    model_config = ConfigDict(extra="forbid")

    run_at: datetime
    interval_seconds: int

    def validate_semantics(self) -> None:
        """フィールド間の整合性を検証する（型検証だけでは表現できない制約）."""
        if self.run_at.tzinfo is None:
            raise ValueError("run_at must be timezone-aware")
        if self.interval_seconds != DispatcherService.ONE_SHOT and self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be -1 or a positive integer")


class DispatcherMaintenance:
    """Dispatch を Markdown ファイルへ書き出し・取り込みする保守処理."""

    def __init__(self, dispatch_entity: type, sa_engine: AsyncEngine, root: Path) -> None:
        """予約を dispatch_entity のテーブル（sa_engine 経由）で管理し、root 配下へ書き出す."""
        self._entity = dispatch_entity
        self._engine = sa_engine
        self._root = root

    @staticmethod
    def _render_dispatch(dispatch: Dispatch) -> str:
        """Dispatch を、予約ファイルの schema に従う Markdown 文字列へ変換する."""
        front_matter = yaml.safe_dump(
            {"run_at": dispatch.run_at.isoformat(), "interval_seconds": dispatch.interval_seconds},
            sort_keys=False,
        )
        return f"{FRONT_MATTER_DELIMITER}{front_matter}{FRONT_MATTER_DELIMITER}{dispatch.prompt}"

    @staticmethod
    def _parse_dispatch(text: str, agent_id: str, dispatch_id: str) -> Dispatch:
        """予約ファイルの Markdown 文字列を Dispatch へ変換する.

        agent_id・dispatch_id はファイルパスから抽出した値を呼び出し元から受け取る
        """
        raw_front_matter, prompt = MarkdownLoader.split_front_matter(text)
        if raw_front_matter is None:
            raise ValueError("front matter delimiter '---' not found")

        try:
            raw = yaml.safe_load(raw_front_matter)
        except yaml.YAMLError as e:
            raise ValueError(f"invalid YAML front matter: {e}") from e

        try:
            front_matter = _DispatchFrontMatter.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"invalid front matter: {e}") from e
        front_matter.validate_semantics()

        return Dispatch(
            dispatch_id=dispatch_id,
            agent_id=agent_id,
            prompt=prompt,
            interval_seconds=front_matter.interval_seconds,
            run_at=front_matter.run_at,
        )

    async def dump(self, agent_id: str) -> list[Path]:
        """指定 agent_id の予約を Markdown ファイルへ書き出し、書き出したパス一覧を返す."""
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.scalars(
                    select(self._entity)
                    .where(self._entity.agent_id == agent_id)
                    .order_by(self._entity.dispatch_id)
                )
            ).all()

        agent_dir = self._root / agent_id
        if agent_dir.exists():
            shutil.rmtree(agent_dir)
        agent_dir.mkdir(parents=True)

        paths = []
        for row in rows:
            dispatch = Dispatch(
                row.dispatch_id, row.agent_id, row.prompt, row.interval_seconds, row.run_at
            )
            path = agent_dir / f"{dispatch.dispatch_id}.md"
            path.write_text(self._render_dispatch(dispatch), encoding="utf-8")
            paths.append(path)
        return paths

    async def load(self, agent_id: str) -> list[Dispatch]:
        """指定 agent_id の予約を Markdown ファイルから読み、DB を洗い替える.

        ファイルが0件でも指定 agent_id の全予約削除として扱う。存在しない
        agent_id ディレクトリは誤操作としてエラーにする。
        """
        agent_dir = self._root / agent_id
        if not agent_dir.exists():
            raise ValueError(f"directory not found: {agent_dir}")

        dispatches = []
        for path in sorted(agent_dir.glob("*.md")):
            uuid.UUID(path.stem)
            dispatches.append(
                self._parse_dispatch(path.read_text(encoding="utf-8"), agent_id, path.stem)
            )
        dispatch_ids = [d.dispatch_id for d in dispatches]
        if len(dispatch_ids) != len(set(dispatch_ids)):
            raise ValueError(f"duplicate dispatch_id in {agent_dir}")

        async with AsyncSession(self._engine) as session:
            await session.execute(delete(self._entity).where(self._entity.agent_id == agent_id))
            for dispatch in dispatches:
                session.add(
                    self._entity(
                        dispatch_id=dispatch.dispatch_id,
                        agent_id=dispatch.agent_id,
                        prompt=dispatch.prompt,
                        interval_seconds=dispatch.interval_seconds,
                        run_at=dispatch.run_at,
                    )
                )
            await session.commit()
        return dispatches
