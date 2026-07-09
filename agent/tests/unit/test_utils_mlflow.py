from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import Message

from assistant_agent.utils.mlflow import LangGraphChatAgent, LangGraphResponsesAgent


class TestLangGraphResponsesAgent:
    def test_predict(self):
        """推論 (同期) を正しく呼び出せるか確認.

        観点2:
            updates/messages 混在出力のうち、updates モードの done イベントのみが
            output に収集される
            - updates モード → response.output_item.done → output に収集される
            - messages モード → response.output_text.delta → output に含まれない
        """
        # 試験準備
        m_agent = MagicMock()
        m_agent.stream.return_value = iter(self._make_mixed_stream())
        request = ResponsesAgentRequest(input=[Message(role="user", content="hello")])

        # 試験実施
        m_context = MagicMock()
        wrapper = LangGraphResponsesAgent(m_agent, m_context)
        result = wrapper.predict(request)

        # 観点1: 入力がエージェントに正しく渡される
        m_agent.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"recursion_limit": 100},
            context=m_context,
            # 暫定対処
            # stream_mode=["updates", "messages"],
            stream_mode=["updates"],
        )
        # 観点2: 出力が ResponsesAgentStreamEvent に変換される
        assert len(result.output) == 1
        assert result.output[0].type == "message"

    def test_predict_stream(self):
        """推論 (ストリーム) を正しく呼び出せるか確認.

        観点1:
            Message インスタンスのリストを格納した ResponsesAgentRequest が
            エージェントに変換されて渡される
        観点2: updates/messages 混在出力が ResponsesAgentStreamEvent として yield される
            - updates モード: AIMessage → response.output_item.done
            - messages モード: AIMessageChunk → response.output_text.delta
        """
        # 試験準備
        m_agent = MagicMock()
        m_agent.stream.return_value = iter(self._make_mixed_stream())
        request = ResponsesAgentRequest(input=[Message(role="user", content="hello")])

        # 試験実施
        m_context = MagicMock()
        wrapper = LangGraphResponsesAgent(m_agent, m_context)
        events = list(wrapper.predict_stream(request))

        # 観点1: 入力がエージェントに正しく渡される
        m_agent.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"recursion_limit": 100},
            context=m_context,
            # 暫定対処
            # stream_mode=["updates", "messages"],
            stream_mode=["updates"],
        )
        # 観点2: 出力が ResponsesAgentStreamEvent に変換される
        assert all(isinstance(e, ResponsesAgentStreamEvent) for e in events)
        done_events = [e for e in events if e.type == "response.output_item.done"]
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        assert len(done_events) == 1
        assert len(delta_events) == 1
        assert (
            delta_events[0].delta  # pyright: ignore[reportAttributeAccessIssue]
            == "streaming text"
        )

    @staticmethod
    def _make_mixed_stream():
        """updates/messages 混在のストリームデータを返すヘルパー."""
        ai_msg = AIMessage(content="agent response")
        ai_chunk = AIMessageChunk(
            content=[{"type": "text", "text": "streaming text"}], id="chunk-1"
        )
        return [
            ("updates", {"node": {"messages": [ai_msg]}}),
            ("messages", (ai_chunk, None)),
        ]


class TestLangGraphChatAgent:
    def test_predict_01(self):
        """推論 (同期) を正しく呼び出せるか確認.

        観点1: ainvoke が正しい引数で呼び出される
        観点2: 最後の assistant メッセージのみが返ること
        """
        # 試験準備
        m_agent = MagicMock()
        m_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="hello", id="h-1"),
                    AIMessage(content="first response", id="a-1"),
                    ToolMessage(
                        content="tool result", tool_call_id="call-1", name="get_weather", id="t-1"
                    ),
                    AIMessage(content="final response", id="a-2"),
                ]
            }
        )
        messages = [ChatAgentMessage(role="user", content="hello")]

        # 試験実施
        m_context = MagicMock()
        wrapper = LangGraphChatAgent(m_agent, m_context)
        result = wrapper.predict(messages)

        # 観点1
        m_agent.ainvoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"recursion_limit": 100},
            context=m_context,
        )
        # 観点2
        assert len(result.messages) == 1
        assert result.messages[0].role == "assistant"
        assert result.messages[0].content == "final response"

    def test_predict_stream_01(self):
        """推論 (ストリーム) を正しく呼び出せるか確認.

        観点1: stream_mode=["messages"] で呼び出される
        観点2: "updates" モードが正しく動作する
            - model ノードの AIMessage が ChatAgentChunk として yield される
            - tools ノード (ToolMessage) は ChatAgentChunk として yield される
            - content 空の AIMessageChunk はスキップされる
        観点3: "messages" モードが正しく動作する
            - AIMessageChunk がテキストデルタに変換される
        """
        # 試験準備
        m_agent = MagicMock()
        m_agent.stream.return_value = iter(self._make_mixed_stream())
        messages = [ChatAgentMessage(role="user", content="hello")]

        # 試験実施
        m_context = MagicMock()
        wrapper = LangGraphChatAgent(m_agent, m_context)
        chunks = list(wrapper.predict_stream(messages))

        # 観点1
        m_agent.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"recursion_limit": 100},
            stream_mode=["messages"],
            context=m_context,
        )
        # 観点2-5: 3件が yield される
        assert len(chunks) == 3
        assert all(isinstance(c, ChatAgentChunk) for c in chunks)
        # 観点2: "updates" モードが正しく動作する
        assert chunks[0].delta.content == "agent response"
        assert chunks[1].delta.content == "tool result"
        # 観点3: "messages" モードが正しく動作する
        assert chunks[2].delta.content == "streaming text"
        # 観点3: content 空の AIMessageChunk はスキップ → 3件のみ

    def test_predict_stream_02(self):
        """推論 (ストリーム) が正しく呼び出せるか確認 (model ノードに tool_calls がある場合).

        観点1:
            "updates" モードの model ノードが tool_calls を持つ場合、tool_calls のみを
            含む ChatAgentChunk が yield される
        観点2: tool_calls チャンクの content は空文字になること
        """
        # 試験準備
        m_agent = MagicMock()
        m_agent.stream.return_value = iter(self._make_tool_call_stream())
        messages = [ChatAgentMessage(role="user", content="hello")]

        # 試験実施
        m_context = MagicMock()
        wrapper = LangGraphChatAgent(m_agent, m_context)
        chunks = list(wrapper.predict_stream(messages))

        # 観点1: tool_calls チャンクが 1件 yield される
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatAgentChunk)
        assert chunks[0].delta.tool_calls is not None
        assert chunks[0].delta.tool_calls[0].function.name == "get_weather"
        # 観点2: content は None
        assert chunks[0].delta.content == ""

    @staticmethod
    def _make_mixed_stream():
        """updates/messages 混在のストリームデータを返すヘルパー."""
        tool_msg = ToolMessage(
            content="tool result", tool_call_id="call-1", name="get_weather", id="tool-1"
        )
        ai_chunk = AIMessageChunk(
            content=[{"type": "text", "text": "streaming text"}], id="chunk-1"
        )
        empty_chunk = AIMessageChunk(content=[], id="chunk-2")
        return [
            ("updates", {"model": {"messages": [AIMessage(content="agent response")]}}),
            ("updates", {"tools": {"messages": [tool_msg]}}),
            ("messages", (ai_chunk, None)),
            ("messages", (empty_chunk, None)),
        ]

    @staticmethod
    def _make_tool_call_stream():
        """Model ノードが tool_calls を持つ AIMessage を返すストリームデータ."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "get_weather", "args": {"city": "Tokyo"}}],
            id="ai-1",
        )
        return [
            ("updates", {"model": {"messages": [ai_msg]}}),
        ]
