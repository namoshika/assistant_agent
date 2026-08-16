import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import discord
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from assistant_agent.utils.absclass import ActiveEmitter, AgentInvocation
from assistant_agent.utils.context import ContextRegistry

DISCORD_CHANNEL_TMPL = """\
# Discord Channel (channel_id: {channel_id}, guild_id: {guild_id})
## Guideline
受信した Discord チャネルの新着ポストが来ます。
チャネルには他のメンバーも居るため、新着ポストがあなた宛てとは限りません。
応答基準を満たすか判別し、満たせば応答してください。満たさない場合には絶対に応答しないでください。

1. 新着受信時
    1. 直近投稿を 20 件取得し、直近の文脈を把握。自身に関係の有るものか判別
    2. 関係有る場合: 応答する。応答は必ず受信した Discord サーバーの同じチャンネルへ送信する
    3. 関係ない場合: 応答しない (サーバーには複数人参加しているため、自分宛でない場合は応答禁止)
2. 応答時 (特定の投稿やユーザに対してリアクションする場合)
    1. 対象の投稿が直近10件以内の投稿の場合: ツール discord_send_message で投稿
    2. 対象の投稿が直近10件より前の投稿の場合: ツール discord_reply_message を使用
    3. 対象の相手を指名した投稿の場合: ツール discord_mention_user を使用

応答するかの判断基準

- 応答する
  - 自身の名前を呼ばれている
  - コンテキストウィンドウ内で過去に自身も言及している会話
  - コンテキストウィンドウ内で投稿受信から5分以上誰からも反応を得ていない投稿
- 応答しない
  - 他人の名前を呼ばれている

応答する場合、相槌や結論の反復だけで終わらせず、自分の視点・関心・疑問のいずれかを添えて話題を1歩広げること。
可能であれば相手が反応しやすい問いかけや話の続きを残すこと。

## Output Format
行動後、以下の形式で出力してください。

```
# Activity Log

## Observe
(観測した事実)

## Understand
(現在の状況・目的)

## Consider
(判断・方針)

## Act
(実行したこと)

## Verify
(結果・確認)

## Report
(ユーザーへの報告)
```

## Received Message
{messages}
"""

DISCORD_MESSAGE_TMPL = """\
## Discord Post

```yaml
message_id (メッセージ ID): {message_id}
author_id (投稿者の Discord ユーザー ID): {author_id}
author_name (投稿者の表示名): {author_name}
created_at (投稿日時（JST)): {created_at}
is_owned (自分（Bot 自身）の投稿かどうか): {is_owned}
```

{content}
"""


@dataclass(frozen=True)
class DiscordIncomingMessage:
    """DiscordService が受信した新着メッセージ（discord.py の型に依存しない受け渡し用）."""

    guild_id: int
    channel_id: int
    message_id: int
    author_id: int
    author_name: str
    created_at: str
    is_owned: bool
    content: str

    def __str__(self) -> str:
        """文字列化."""
        return DISCORD_MESSAGE_TMPL.format(
            message_id=self.message_id,
            author_id=self.author_id,
            author_name=self.author_name,
            created_at=self.created_at,
            is_owned=self.is_owned,
            content=self.content,
        )


class DiscordService:
    """discord.py の Bot クライアントに対する実処理（送信・取得・接続制御）をまとめたクラス."""

    def __init__(self, token: str | None, *, ignore_own_messages: bool = True):
        """token（Bot トークン）を保持し discord.Client を構成する（token は None も許容）.

        ignore_own_messages は on_message() が Bot 自身の発言を無視するかどうかを制御する
        （デフォルト True。結合テストが自分自身の投稿で受信を検証する用途にのみ False を渡す）。
        """
        self._token = token
        self._ignore_own_messages = ignore_own_messages
        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._task: asyncio.Task[None] | None = None
        self._message_handler: Callable[[DiscordIncomingMessage], None] | None = None
        self._client.event(self.on_message)

    def set_message_handler(self, handler: Callable[[DiscordIncomingMessage], None]) -> None:
        """新着メッセージの受け取り先を登録する（登録は1つのみ。再登録は上書き）."""
        self._message_handler = handler

    async def on_message(self, message: discord.Message) -> None:
        """discord.py のイベントディスパッチから呼ばれる（メソッド名は on_message で固定）."""
        if self._message_handler is None:
            return
        if self._ignore_own_messages and message.author == self._client.user:
            return
        if message.guild is None:
            return
        self._message_handler(self._to_incoming_message(message))

    @property
    def guild_ids(self) -> list[int]:
        """参加中サーバー ID の一覧を取得する."""
        return [guild.id for guild in self._client.guilds]

    async def wait_until_ready(self) -> None:
        """discord.py の接続完了を待つ（結合テストが起動直後の疎通待ちに使う）."""
        await self._client.wait_until_ready()

    def start(self) -> None:
        """接続を開始する（起動済みなら何もしない）."""
        if self._task is not None and not self._task.done():
            return
        if self._token is None:
            raise RuntimeError("AA_DISCORD_BOT_TOKEN が未設定のため Discord へ接続できません。")
        self._task = asyncio.create_task(self._client.start(self._token))

    def stop(self) -> None:
        """接続を終了する."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def get_messages(self, channel_id: int, limit: int = 20) -> list[DiscordIncomingMessage]:
        """指定チャンネルの直近メッセージ一覧を取得する."""
        channel = self._client.get_channel(channel_id)
        assert isinstance(channel, discord.TextChannel)
        return [self._to_incoming_message(msg) async for msg in channel.history(limit=limit)]

    async def send_message(
        self, channel_id: int, content: str, *, suppress_embeds: bool = False
    ) -> None:
        """指定チャンネルへメンションなしでメッセージを投稿する."""
        channel = self._client.get_channel(channel_id)
        assert isinstance(channel, discord.TextChannel)
        await channel.send(content, suppress_embeds=suppress_embeds)

    async def reply_message(
        self, channel_id: int, message_id: int, content: str, *, suppress_embeds: bool = False
    ) -> None:
        """指定メッセージへ返信する."""
        channel = self._client.get_channel(channel_id)
        assert isinstance(channel, discord.TextChannel)
        message = await channel.fetch_message(message_id)
        await message.reply(content, suppress_embeds=suppress_embeds)

    async def mention_user(
        self, channel_id: int, user_id: int, content: str, *, suppress_embeds: bool = False
    ) -> None:
        """指定チャンネルへユーザーへのメンション付きメッセージを投稿する."""
        channel = self._client.get_channel(channel_id)
        assert isinstance(channel, discord.TextChannel)
        await channel.send(f"<@{user_id}> {content}", suppress_embeds=suppress_embeds)

    def list_channels(self) -> list[dict[str, int]]:
        """参加中のテキストチャンネル一覧（サーバー ID・チャンネル ID）を取得する."""
        return [
            {"guild_id": ch.guild.id, "channel_id": ch.id}
            for ch in self._client.get_all_channels()
            if isinstance(ch, discord.TextChannel)
        ]

    def _to_incoming_message(self, message: discord.Message) -> DiscordIncomingMessage:
        """discord.Message を discord.py 非依存の DiscordIncomingMessage へ変換する."""
        assert message.guild is not None
        is_owned = self._client.user is not None and message.author.id == self._client.user.id
        return DiscordIncomingMessage(
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            created_at=message.created_at.astimezone(ZoneInfo("Asia/Tokyo")).isoformat(),
            content=message.content,
            is_owned=is_owned,
        )


@ContextRegistry.register("discord_service")
def build(**_: Any) -> DiscordService:
    """DiscordService を生成する（AA_DISCORD_BOT_TOKEN 未設定時は未接続状態のまま返す）."""
    logger = logging.getLogger(__name__)
    token = os.getenv("AA_DISCORD_BOT_TOKEN")
    if token is None:
        logger.warning("AA_DISCORD_BOT_TOKEN が未設定のため、Discord 機能は利用できません。")
    return DiscordService(token=token)


class DiscordChannel(ActiveEmitter):
    """DiscordService の発信を Emitter として配線するクラス（discord.py 非依存）."""

    def __init__(self, service: DiscordService):
        """service（DiscordService）を受け取り、新着メッセージのハンドラを登録する."""
        super().__init__()
        self._service = service
        self._service.set_message_handler(self._on_message)

    def start(self) -> None:
        """DiscordService への接続開始を委譲する."""
        self._service.start()

    def stop(self) -> None:
        """DiscordService への接続終了を委譲する."""
        self._service.stop()

    def _on_message(self, message: DiscordIncomingMessage) -> None:
        """DiscordService から渡された新着メッセージを AgentInvocation にして emit する."""
        content = PromptTemplate.from_template(DISCORD_CHANNEL_TMPL).format(
            guild_id=message.guild_id,
            channel_id=message.channel_id,
            messages=message,
        )
        self.emit(
            AgentInvocation(
                input={"messages": [HumanMessage(content=content)]},
                context={"request_id": str(uuid.uuid7())},
            )
        )
