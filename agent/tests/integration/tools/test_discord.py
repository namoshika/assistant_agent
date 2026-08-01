from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

import assistant_agent.tools.discord as tools
from assistant_agent.services.discord import DiscordIncomingMessage, DiscordService
from assistant_agent.tools.discord import DiscordContext


@pytest.mark.integration
async def test_discord_get_messages_01(llm: BaseChatModel) -> None:
    """実際の LLM から discord_get_messages を呼び出せるか確認.

    観点1: LLM が channel_id を含む tool call を生成し get_messages が呼ばれる
        （装飾的な書式は試験範囲外とする）
    観点2: ToolMessage.artifact が get_messages() の返り値と一致すること
    """
    # 試験準備
    messages = [
        DiscordIncomingMessage(
            guild_id=999,
            channel_id=123456789,
            message_id=1,
            author_id=222,
            author_name="Some User",
            is_owned=False,
            created_at="2026-08-01T21:34:56+09:00",
            content="直近の投稿",
        )
    ]
    m_service = MagicMock(spec=DiscordService)
    m_service.get_messages = AsyncMock(return_value=messages)
    agent = create_agent(
        model=llm, tools=[tools.discord_get_messages], context_schema=DiscordContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="チャンネル ID 123456789 の直近のメッセージを取得して")
            ]
        },
        context=cast(DiscordContext, {"discord_service": m_service}),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "直近の投稿" in tool_msg.content
    assert m_service.get_messages.call_args.args[0] == 123456789
    # 観点2
    assert tool_msg.artifact is messages


@pytest.mark.integration
async def test_discord_send_message_01(llm: BaseChatModel) -> None:
    """実際の LLM から discord_send_message を呼び出せるか確認.

    観点1: LLM が channel_id・content を含む tool call を生成し send_message が呼ばれる
    """
    # 試験準備
    m_service = MagicMock(spec=DiscordService)
    m_service.send_message = AsyncMock(return_value=None)
    agent = create_agent(
        model=llm, tools=[tools.discord_send_message], context_schema=DiscordContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="チャンネル ID 123456789 へ「おはよう」と投稿して")]},
        context=cast(DiscordContext, {"discord_service": m_service}),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "sent"
    assert m_service.send_message.call_args.args[0] == 123456789


@pytest.mark.integration
async def test_discord_mention_user_01(llm: BaseChatModel) -> None:
    """実際の LLM から discord_mention_user を呼び出せるか確認.

    観点1: LLM が channel_id・user_id・content を含む tool call を生成し mention_user が呼ばれる
    """
    # 試験準備
    m_service = MagicMock(spec=DiscordService)
    m_service.mention_user = AsyncMock(return_value=None)
    agent = create_agent(
        model=llm, tools=[tools.discord_mention_user], context_schema=DiscordContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="チャンネル ID 123456789 でユーザー ID 987654321 へ"
                    "「おはよう」とメンション付きで投稿して"
                )
            ]
        },
        context=cast(DiscordContext, {"discord_service": m_service}),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "sent"
    assert m_service.mention_user.call_args.args[0] == 123456789
    assert m_service.mention_user.call_args.args[1] == 987654321


@pytest.mark.integration
async def test_discord_reply_message_01(llm: BaseChatModel) -> None:
    """実際の LLM から discord_reply_message を呼び出せるか確認.

    観点1: LLM が channel_id・message_id・content を含む tool call を生成し
        reply_message が呼ばれる
    """
    # 試験準備
    m_service = MagicMock(spec=DiscordService)
    m_service.reply_message = AsyncMock(return_value=None)
    agent = create_agent(
        model=llm, tools=[tools.discord_reply_message], context_schema=DiscordContext
    )

    # 試験実施
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="チャンネル ID 123456789 のメッセージ ID 555555555 へ"
                    "「承知しました」と返信して"
                )
            ]
        },
        context=cast(DiscordContext, {"discord_service": m_service}),
    )

    # 結果検証
    # 観点1
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "sent"
    assert m_service.reply_message.call_args.args[0] == 123456789
    assert m_service.reply_message.call_args.args[1] == 555555555
