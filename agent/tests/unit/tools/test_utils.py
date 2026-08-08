import urllib.error
from email.message import Message
from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pytest_mock import MockerFixture

import assistant_agent.tools.utils as tools


class _FakeChatModel(GenericFakeChatModel):
    """bind_tools() をサポートするダミーチャットモデル."""

    def bind_tools(self, tools, **_):
        return self


def test_web_fetch_01(mocker: MockerFixture):
    """指定 URL に GET リクエストしレスポンスを返せるか確認.

    観点1: urlopen に渡す Request の User-Agent が Chrome の値であること
    観点2: 戻り値に URL・ステータスコード・レスポンスヘッダー・本文が含まれること
    """
    # 試験準備
    headers = Message()
    headers["Content-Type"] = "text/plain; charset=utf-8"
    m_resp = MagicMock()
    m_resp.status = 200
    m_resp.headers = headers
    m_resp.read.return_value = b"response body"
    m_resp.__enter__.return_value = m_resp
    m_urlopen = mocker.patch("urllib.request.urlopen", return_value=m_resp)

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_fetch",
                "args": {"url": "https://example.com/feed.rss"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.web_fetch)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="RSS を取得して")]})

    # 結果検証
    # 観点1
    request = m_urlopen.call_args.args[0]
    assert "Chrome" in request.headers["User-agent"]
    # 観点2
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "https://example.com/feed.rss" in tool_msg.content
    assert "200" in tool_msg.content
    assert "response body" in tool_msg.content


def test_web_fetch_02(mocker: MockerFixture):
    """Content-Type に charset が無い場合に UTF-8 でデコードされるか確認.

    観点: 日本語（UTF-8）の本文が正しくデコードされて返る
    """
    # 試験準備
    headers = Message()
    headers["Content-Type"] = "text/plain"
    m_resp = MagicMock()
    m_resp.status = 200
    m_resp.headers = headers
    m_resp.read.return_value = "日本語本文".encode()
    m_resp.__enter__.return_value = m_resp
    mocker.patch("urllib.request.urlopen", return_value=m_resp)

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_fetch",
                "args": {"url": "https://example.com/"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.web_fetch)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="取得して")]})

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "日本語本文" in tool_msg.content


def test_web_fetch_03(mocker: MockerFixture):
    """HTTPError（4xx/5xx 応答）発生時に例外を送出せずエラー内容を返せるか確認.

    観点: ステータスコード・レスポンスヘッダー・エラーレスポンス本文が含まれること
    """
    # 試験準備
    headers = Message()
    headers["Content-Type"] = "text/plain; charset=utf-8"
    ex = urllib.error.HTTPError(
        url="https://example.com/notfound",
        code=404,
        msg="NOT FOUND",
        hdrs=headers,
        fp=None,
    )
    mocker.patch.object(ex, "read", return_value=b"not found body")
    mocker.patch("urllib.request.urlopen", side_effect=ex)

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_fetch",
                "args": {"url": "https://example.com/notfound"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.web_fetch)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="取得して")]})

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "404" in tool_msg.content
    assert "not found body" in tool_msg.content


def test_web_fetch_04(mocker: MockerFixture):
    """HTTPError のレスポンス本文が空の場合に str(ex) へフォールバックするか確認.

    観点: エラーレスポンス本文が空の際、例外メッセージが本文として返ること
    """
    # 試験準備
    headers = Message()
    ex = urllib.error.HTTPError(
        url="https://example.com/notfound",
        code=404,
        msg="NOT FOUND",
        hdrs=headers,
        fp=None,
    )
    mocker.patch.object(ex, "read", return_value=b"")
    mocker.patch("urllib.request.urlopen", side_effect=ex)

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_fetch",
                "args": {"url": "https://example.com/notfound"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.web_fetch)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="取得して")]})

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "HTTP Error 404" in tool_msg.content


def test_web_fetch_05(mocker: MockerFixture):
    """URLError（接続不可等）発生時に例外を送出せずエラー内容を返せるか確認.

    観点: ステータスコード・ヘッダーを含まず、エラー内容のみが返ること
    """
    # 試験準備
    ex = urllib.error.URLError(reason="Name or service not known")
    mocker.patch("urllib.request.urlopen", side_effect=ex)

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_fetch",
                "args": {"url": "https://this-domain-does-not-exist.invalid/"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    agent = _make_agent(ai_msg, tools.web_fetch)

    # 試験実施
    result = agent.invoke({"messages": [HumanMessage(content="取得して")]})

    # 結果検証
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Name or service not known" in tool_msg.content
    assert "status: error" in tool_msg.content


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
