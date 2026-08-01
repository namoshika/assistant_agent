from unittest.mock import AsyncMock, MagicMock

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import assistant_agent.tools.discord as tools
from assistant_agent.services.discord import DiscordIncomingMessage, DiscordService


class _FakeChatModel(GenericFakeChatModel):
    """bind_tools() をサポートするダミーチャットモデル."""

    def bind_tools(self, tools, **_):
        return self


async def test_discord_get_messages_01():
    """指定チャンネルの直近メッセージ一覧を取得できるか確認.

    観点1（R013・R014）: service.get_messages(channel_id, limit) が正しい引数で呼ばれること
    観点2: 取得した各メッセージの内容が ToolMessage.content に含まれること
        （装飾的な書式は試験範囲外とする）
    観点3: ToolMessage.artifact が service.get_messages() の返り値と一致すること
    """
    # 試験準備
    messages = [
        DiscordIncomingMessage(
            guild_id=999,
            channel_id=111,
            message_id=1,
            author_id=222,
            author_name="Some User",
            is_owned=False,
            created_at="2026-08-01T21:34:56+09:00",
            content="こんにちは",
        ),
        DiscordIncomingMessage(
            guild_id=999,
            channel_id=111,
            message_id=2,
            author_id=222,
            author_name="Some User",
            is_owned=False,
            created_at="2026-08-01T21:35:00+09:00",
            content="よろしくお願いします",
        ),
    ]
    m_service = MagicMock(spec=DiscordService)
    m_service.get_messages = AsyncMock(return_value=messages)
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "discord_get_messages",
                "args": {"channel_id": "111", "limit": 5},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.discord_get_messages)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="直近のメッセージを見せて")]},
        context={"discord_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.get_messages.assert_called_once_with(111, 5)
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "こんにちは" in tool_msg.content
    assert "よろしくお願いします" in tool_msg.content
    # 観点3
    assert tool_msg.artifact is messages


async def test_discord_send_message_01():
    """指定チャンネルへメッセージを投稿できるか確認.

    観点1（R013・R014）: service.send_message(channel_id, content) が正しい引数で呼ばれること
    観点2: 投稿完了を示す ToolMessage が返ること
    """
    # 試験準備
    m_service = MagicMock(spec=DiscordService)
    m_service.send_message = AsyncMock()
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "discord_send_message",
                "args": {"channel_id": "111", "content": "hello"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.discord_send_message)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="投稿して")]},
        context={"discord_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.send_message.assert_called_once_with(111, "hello")
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "sent"


async def test_discord_mention_user_01():
    """指定チャンネルでユーザーへメンション付き投稿ができるか確認.

    観点1（R013・R014）: service.mention_user(channel_id, user_id, content) が
        正しい引数で呼ばれること
    観点2: 投稿完了を示す ToolMessage が返ること
    """
    # 試験準備
    m_service = MagicMock(spec=DiscordService)
    m_service.mention_user = AsyncMock()
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "discord_mention_user",
                "args": {"channel_id": "111", "user_id": "222", "content": "hello"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.discord_mention_user)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="メンションして投稿して")]},
        context={"discord_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.mention_user.assert_called_once_with(111, 222, "hello")
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "sent"


async def test_discord_reply_message_01():
    """指定メッセージへ返信できるか確認.

    観点1（R013・R014）: service.reply_message(channel_id, message_id, content) が
        正しい引数で呼ばれること
    観点2: 投稿完了を示す ToolMessage が返ること
    """
    # 試験準備
    m_service = MagicMock(spec=DiscordService)
    m_service.reply_message = AsyncMock()
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "discord_reply_message",
                "args": {"channel_id": "111", "message_id": "333", "content": "hello"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.discord_reply_message)

    # 試験実施
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="返信して")]},
        context={"discord_service": m_service},
    )

    # 結果検証
    # 観点1
    m_service.reply_message.assert_called_once_with(111, 333, "hello")
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "sent"


def _make_agent(tool_calls_msg: AIMessage, *tools):
    """テスト用エージェントを生成するヘルパー."""
    fake_llm = _FakeChatModel(
        messages=iter(
            [
                tool_calls_msg,
                AIMessage(content="完了しました。"),
            ]
        )
    )
    return create_agent(model=fake_llm, tools=list(tools))
