import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest

from assistant_agent.services.discord import DiscordService


@pytest.fixture
async def discord_service() -> AsyncIterator[DiscordService]:
    """実際の Discord Bot トークンで接続した DiscordService.

    AA_DISCORD_BOT_TOKEN・AA_DISCORD_CHANNEL_ID 環境変数が必要。
    """
    token = os.environ.get("AA_DISCORD_BOT_TOKEN")
    if not token:
        pytest.fail("AA_DISCORD_BOT_TOKEN が未設定のため失敗")

    # ignore_own_messages=False: test_on_message_01 が自分自身（Bot）の投稿で
    # 受信を検証するため。send_message/get_messages/list_channels には影響しない
    service = DiscordService(token=token, ignore_own_messages=False)
    service.start()
    # start() は asyncio.create_task() でタスクを積むだけで即座に返るため、
    # そのタスクへ実行機会を1度渡してから待つ（渡さないと discord.py 側の
    # 内部状態（_ready）が未初期化のまま RuntimeError になる）
    await asyncio.sleep(0)
    await asyncio.wait_for(service.wait_until_ready(), timeout=30)
    yield service
    service.stop()


@pytest.fixture
def discord_channel_id() -> int:
    """結合テストで投稿・取得対象とするテキストチャンネル ID."""
    channel_id = os.environ.get("AA_DISCORD_CHANNEL_ID")
    if not channel_id:
        pytest.fail("AA_DISCORD_CHANNEL_ID が未設定のため失敗")
    return int(channel_id)


@pytest.mark.integration
async def test_send_and_get_messages_01(discord_service: DiscordService, discord_channel_id: int):
    """実際の Discord サーバーへメッセージを送信し、直近メッセージ一覧から取得できるか確認.

    観点1（R002・R003）: send_message() で投稿した内容が get_messages() の結果に含まれること
    """
    # 試験準備
    content = f"integration-test-{uuid.uuid4().hex[:8]}"

    # 試験実施
    await discord_service.send_message(discord_channel_id, content)
    messages = await discord_service.get_messages(discord_channel_id, limit=5)

    # 結果検証
    # 観点1
    assert any(m.content == content for m in messages)


@pytest.mark.integration
async def test_list_channels_01(discord_service: DiscordService, discord_channel_id: int):
    """参加中のテキストチャンネル一覧に対象チャンネルが含まれるか確認.

    観点1（R006）: list_channels() の結果に discord_channel_id が含まれること
        （MESSAGE_CONTENT 特権インテントが有効でないと get_messages() の content が
        取得できないため、本テストの前提として R006 の有効化を間接確認する）
    """
    # 試験実施
    channels = discord_service.list_channels()

    # 結果検証
    # 観点1
    assert any(ch["channel_id"] == discord_channel_id for ch in channels)


@pytest.mark.integration
async def test_on_message_01(discord_service: DiscordService, discord_channel_id: int):
    """新着メッセージをリアルタイムに受信できるか確認.

    観点1（R011）: set_message_handler() でハンドラを登録すると、
        投稿したメッセージの content が DiscordIncomingMessage 経由で受信できること
    """
    # 試験準備
    received: asyncio.Queue = asyncio.Queue()
    discord_service.set_message_handler(received.put_nowait)

    content = f"integration-test-{uuid.uuid4().hex[:8]}"

    # 試験実施
    await discord_service.send_message(discord_channel_id, content)
    message = await asyncio.wait_for(received.get(), timeout=30)

    # 結果検証
    # 観点1
    assert message.content == content
