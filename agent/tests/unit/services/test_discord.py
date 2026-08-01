import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from pytest_mock import MockerFixture

from assistant_agent.services.discord import DiscordChannel, DiscordIncomingMessage, DiscordService
from assistant_agent.utils.absclass import AgentInvocation, Receiver


class TestDiscordService:
    async def test_get_messages_01(self, mocker: MockerFixture):
        """get_messages() が直近メッセージを DiscordIncomingMessage 化して返すことを確認.

        観点1（R002）: history() で取得した discord.Message 群が DiscordIncomingMessage
            （on_message() と同様の変換）のリストとして返ること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        bot_user = MagicMock(spec=discord.ClientUser, id=999)
        client.user = bot_user
        service = DiscordService(token="dummy-token")
        raw_message = MagicMock(spec=discord.Message)
        raw_message.guild = MagicMock(id=111)
        raw_message.channel = MagicMock(id=333)
        raw_message.id = 444
        raw_message.author = MagicMock(spec=discord.Member, id=222, display_name="Other User")
        raw_message.content = "hello"
        raw_message.created_at = datetime(2026, 8, 1, 12, 34, 56, tzinfo=UTC)

        async def _history(limit: int):
            yield raw_message

        channel = MagicMock(spec=discord.TextChannel)
        channel.history = _history
        client.get_channel.return_value = channel

        # 試験実施
        result = await service.get_messages(111, limit=5)

        # 結果検証
        # 観点1
        assert result == [
            DiscordIncomingMessage(
                guild_id=111,
                channel_id=333,
                message_id=444,
                author_id=222,
                author_name="Other User",
                is_owned=False,
                created_at="2026-08-01T21:34:56+09:00",
                content="hello",
            )
        ]
        client.get_channel.assert_called_once_with(111)

    async def test_get_messages_02(self, mocker: MockerFixture):
        """指定チャンネルがテキストチャンネルでない場合、AssertionError を送出することを確認.

        観点1（R002）: get_channel() の戻り値が discord.TextChannel でなければ例外送出されること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        client.get_channel.return_value = MagicMock(spec=discord.VoiceChannel)

        # 試験実施・結果検証
        # 観点1
        with pytest.raises(AssertionError):
            await service.get_messages(111)

    async def test_send_message_01(self, mocker: MockerFixture):
        """send_message() がメンションなしで channel.send() を呼ぶことを確認.

        観点1（R003）: channel.send(content) がメンションなしで呼ばれること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        client.get_channel.return_value = channel

        # 試験実施
        await service.send_message(111, "hello")

        # 結果検証
        # 観点1
        channel.send.assert_called_once_with("hello")

    async def test_mention_user_01(self, mocker: MockerFixture):
        """mention_user() がメンション形式の文字列を送信することを確認.

        観点1（R004）: channel.send() の引数に <@user_id> を含むメンション形式の文字列が渡ること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        client.get_channel.return_value = channel

        # 試験実施
        await service.mention_user(111, 222, "hello")

        # 結果検証
        # 観点1
        channel.send.assert_called_once_with("<@222> hello")

    async def test_reply_message_01(self, mocker: MockerFixture):
        """reply_message() が指定メッセージへ reply() することを確認.

        観点1（R005）: channel.fetch_message() で取得したメッセージの reply() が呼ばれること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        message = MagicMock(spec=discord.Message)
        message.reply = AsyncMock()
        channel = MagicMock(spec=discord.TextChannel)
        channel.fetch_message = AsyncMock(return_value=message)
        client.get_channel.return_value = channel

        # 試験実施
        await service.reply_message(111, 333, "hello")

        # 結果検証
        # 観点1
        channel.fetch_message.assert_called_once_with(333)
        message.reply.assert_called_once_with("hello")

    async def test_on_message_01(self, mocker: MockerFixture):
        """on_message() が自分自身の発言・DM を無視し、それ以外はハンドラへ変換して渡すことを確認.

        観点1（R011）: 他ユーザーによるサーバー内メッセージを受けると、登録済みハンドラへ
            DiscordIncomingMessage（各値が文字列化されたもの。author_name は表示名、
            is_owned は False、created_at は JST の ISO8601 文字列）が渡されること
        観点2（R011）: 自分自身（bot）の発言はハンドラが呼ばれないこと
        観点3（R011）: message.guild が None（DM）のメッセージはハンドラが呼ばれないこと
        観点4（R011）: ハンドラ未登録時に呼ばれても例外にならないこと
        観点5（R011）: ignore_own_messages=False で構築すると、自分自身（bot）の発言でも
            ハンドラ（is_owned=True）が呼ばれること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        bot_user = MagicMock(spec=discord.ClientUser, id=999)
        client.user = bot_user
        service = DiscordService(token="dummy-token")
        handler = MagicMock()
        service.set_message_handler(handler)

        other_message = MagicMock(spec=discord.Message)
        other_message.author = MagicMock(spec=discord.Member, id=222, display_name="Other User")
        other_message.guild = MagicMock(id=111)
        other_message.channel = MagicMock(id=333)
        other_message.id = 444
        other_message.content = "hello"
        other_message.created_at = datetime(2026, 8, 1, 12, 34, 56, tzinfo=UTC)

        # 試験実施
        await service.on_message(other_message)

        # 結果検証
        # 観点1
        handler.assert_called_once_with(
            DiscordIncomingMessage(
                guild_id=111,
                channel_id=333,
                message_id=444,
                author_id=222,
                author_name="Other User",
                is_owned=False,
                created_at="2026-08-01T21:34:56+09:00",
                content="hello",
            )
        )

        # 試験準備: bot 自身の発言
        handler.reset_mock()
        own_message = MagicMock(spec=discord.Message)
        own_message.author = bot_user
        own_message.guild = MagicMock(id=111)

        # 試験実施
        await service.on_message(own_message)

        # 結果検証
        # 観点2
        handler.assert_not_called()

        # 試験準備: DM（guild が None）
        dm_message = MagicMock(spec=discord.Message)
        dm_message.author = MagicMock(spec=discord.User, id=555)
        dm_message.guild = None

        # 試験実施
        await service.on_message(dm_message)

        # 結果検証
        # 観点3
        handler.assert_not_called()

        # 試験準備: ハンドラ未登録
        service_no_handler = DiscordService(token="dummy-token")

        # 試験実施・結果検証
        # 観点4
        await service_no_handler.on_message(other_message)

        # 試験準備: ignore_own_messages=False
        service_allow_own = DiscordService(token="dummy-token", ignore_own_messages=False)
        service_allow_own.set_message_handler(handler)
        handler.reset_mock()

        # 試験実施
        await service_allow_own.on_message(own_message)

        # 結果検証
        # 観点5
        handler.assert_called_once()
        invocation: DiscordIncomingMessage = handler.call_args[0][0]
        assert invocation.is_owned is True

    def test_guild_ids_01(self, mocker: MockerFixture):
        """guild_ids が参加中サーバー ID のリストとして返すことを確認.

        観点1（R012）: client.guilds の各 id がそのままリストとして返ること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        client.guilds = [MagicMock(id=111), MagicMock(id=222)]

        # 試験実施・結果検証
        # 観点1
        assert service.guild_ids == [111, 222]

    async def test_start_stop_01(self, mocker: MockerFixture):
        """start()/stop() が接続開始・終了を制御することを確認.

        観点1（R007）: start() が client.start() を非同期タスクとして起動すること
        観点2（R007）: stop() がそのタスクをキャンセルすること
        """

        # 試験準備
        async def _forever(*_: object) -> None:
            await asyncio.sleep(3600)

        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        client.start = AsyncMock(side_effect=_forever)

        # 試験実施
        service.start()
        await asyncio.sleep(0.01)

        # 結果検証
        # 観点1
        client.start.assert_called_once_with("dummy-token")
        assert service._task is not None
        task = service._task

        # 試験実施
        service.stop()
        await asyncio.sleep(0.01)

        # 結果検証
        # 観点2
        assert task.cancelled()

    async def test_start_02(self, mocker: MockerFixture):
        """Token が未設定（None）のとき start() が RuntimeError を送出することを確認.

        観点1（R007）: client.start() を呼ばず RuntimeError が送出されること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token=None)
        client.start = AsyncMock()

        # 試験実施・結果検証
        # 観点1
        with pytest.raises(RuntimeError):
            service.start()
        client.start.assert_not_called()

    def test_list_channels_01(self, mocker: MockerFixture):
        """list_channels() がテキストチャンネルのみを対象に一覧を返すことを確認.

        観点1（R008）: client.get_all_channels() の戻り値のうち discord.TextChannel のみが
            結果に含まれること
        """
        # 試験準備
        client = mocker.patch("discord.Client").return_value
        service = DiscordService(token="dummy-token")
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.id = 111
        text_channel.guild = MagicMock(id=999)
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        client.get_all_channels.return_value = [text_channel, voice_channel]

        # 試験実施
        result = service.list_channels()

        # 結果検証
        # 観点1
        assert result == [{"guild_id": 999, "channel_id": 111}]


class TestDiscordChannel:
    def test_init_01(self):
        """__init__() が service.set_message_handler() へ自身の on_message を登録することを確認.

        観点1（R011）: service.set_message_handler() が channel.on_message で呼ばれること
        """
        # 試験準備
        service = MagicMock(spec=DiscordService)

        # 試験実施
        channel = DiscordChannel(service=service)

        # 結果検証
        # 観点1
        service.set_message_handler.assert_called_once_with(channel._on_message)

    def test_on_message_01(self):
        """on_message() が DiscordIncomingMessage を AgentInvocation に変換して emit することを確認.

        観点1（R011）: content を持つ AgentInvocation が emit されること。
            guild_id・channel_id・本文が生成された文字列に含まれること
            （装飾的な書式は試験範囲外とする）
        """
        # 試験準備
        service = MagicMock(spec=DiscordService)
        channel = DiscordChannel(service=service)
        received = MagicMock(spec=Receiver)
        channel.receiver = received
        message = DiscordIncomingMessage(
            guild_id=111,
            channel_id=222,
            message_id=333,
            author_id=444,
            author_name="Some User",
            is_owned=False,
            created_at="2026-08-01T21:34:56+09:00",
            content="hello",
        )

        # 試験実施
        channel._on_message(message)

        # 結果検証
        # 観点1
        received.on_received.assert_called_once()
        invocation: AgentInvocation = received.on_received.call_args[0][0]
        content = invocation["input"]["messages"][-1].content
        assert "111" in content
        assert "222" in content
        assert "hello" in content

    def test_start_01(self):
        """start()/stop() が DiscordService への委譲のみであることを確認.

        観点1（R007・R010）: start()/stop() がそれぞれ service.start()/service.stop() を呼ぶこと
        """
        # 試験準備
        service = MagicMock(spec=DiscordService)
        channel = DiscordChannel(service=service)

        # 試験実施
        channel.start()
        channel.stop()

        # 結果検証
        # 観点1
        service.start.assert_called_once()
        service.stop.assert_called_once()
