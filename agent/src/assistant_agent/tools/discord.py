import logging
from typing import TypedDict

from langchain.tools import ToolRuntime, tool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from assistant_agent.services.discord import (
    DISCORD_CHANNEL_TMPL,
    DiscordIncomingMessage,
    DiscordService,
)


class DiscordContext(TypedDict):
    discord_service: DiscordService


class GetMessagesInput(BaseModel):
    channel_id: str = Field(description="取得対象チャンネルの Discord チャンネル ID")
    limit: int = Field(default=20, description="取得する直近メッセージの最大件数")


@tool(args_schema=GetMessagesInput, response_format="content_and_artifact")
async def discord_get_messages(
    channel_id: str,
    limit: int,
    runtime: ToolRuntime[DiscordContext],
) -> tuple[str, list[DiscordIncomingMessage] | Exception]:
    """指定チャンネルの直近メッセージ一覧を取得する.

    各メッセージには次の項目が付与される。
    - author_id: 投稿者の Discord ユーザー ID
    - author_name: 投稿者の表示名
    - is_owned: 自分（Bot 自身）の投稿かどうか
    - created_at: 投稿日時（JST の ISO8601 文字列）
    """
    service = runtime.context["discord_service"]
    try:
        messages = await service.get_messages(int(channel_id), limit)
        content = PromptTemplate.from_template(DISCORD_CHANNEL_TMPL).format(
            desc="指定した Discord チャネルの直近ポスト",
            guild_id=None,
            channel_id=channel_id,
            messages="\n".join(str(m) for m in messages),
        )
        return content, messages
    except Exception as ex:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to get Discord messages")
        return str(ex), ex


@tool
async def discord_send_message(
    channel_id: str,
    content: str,
    runtime: ToolRuntime[DiscordContext],
    suppress_embeds: bool = Field(
        default=False, description="URL のリンクプレビュー（embed）を非表示にするか"
    ),
) -> str:
    """指定チャンネルへメッセージを投稿する. ※この関数を呼び出さない限り、テキスト出力はユーザーに届かない."""  # noqa: E501
    service = runtime.context["discord_service"]
    try:
        await service.send_message(int(channel_id), content, suppress_embeds=suppress_embeds)
        return "Message sent successfully."
    except Exception as ex:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to send Discord message")
        return str(ex)


@tool
async def discord_mention_user(
    channel_id: str,
    user_id: str,
    content: str,
    runtime: ToolRuntime[DiscordContext],
    suppress_embeds: bool = Field(
        default=False, description="URL のリンクプレビュー（embed）を非表示にするか"
    ),
) -> str:
    """指定チャンネルでユーザーへメンション付き投稿をする.

    メンションは本関数が自動で付与するため、content にメンション（例: <@user_id>）を含めないこと。
    含めるとメンションが重複する。
    """
    service = runtime.context["discord_service"]
    try:
        await service.mention_user(
            int(channel_id), int(user_id), content, suppress_embeds=suppress_embeds
        )
        return "Message sent successfully."
    except Exception as ex:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to mention Discord user")
        return str(ex)


@tool
async def discord_reply_message(
    channel_id: str,
    message_id: str,
    content: str,
    runtime: ToolRuntime[DiscordContext],
    suppress_embeds: bool = Field(
        default=False, description="URL のリンクプレビュー（embed）を非表示にするか"
    ),
) -> str:
    """指定メッセージへ返信する."""
    service = runtime.context["discord_service"]
    try:
        await service.reply_message(
            int(channel_id), int(message_id), content, suppress_embeds=suppress_embeds
        )
        return "Message sent successfully."
    except Exception as ex:
        logger = logging.getLogger(__name__)
        logger.exception("Failed to reply to Discord message")
        return str(ex)
