from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import Message

from agent_assistant.utils.mlflow import LangGraphWrapper


class TestLangGraphWrapper:
    def test_predict_stream(self):
        """predict_stream() の動作を検証する.

        観点1: Message インスタンスのリストを格納した ResponsesAgentRequest が
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
        wrapper = LangGraphWrapper(m_agent, m_context)
        events = list(wrapper.predict_stream(request))

        # 観点1: 入力がエージェントに正しく渡される
        m_agent.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"recursion_limit": 100},
            context=m_context,
            stream_mode=["updates", "messages"],
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

    def test_predict(self):
        """predict() の動作を検証する.

        観点2: updates/messages 混在出力のうち、updates モードの done イベントのみが
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
        wrapper = LangGraphWrapper(m_agent, m_context)
        result = wrapper.predict(request)

        # 観点1: 入力がエージェントに正しく渡される
        m_agent.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"recursion_limit": 100},
            context=m_context,
            stream_mode=["updates", "messages"],
        )
        # 観点2: 出力が ResponsesAgentStreamEvent に変換される
        assert len(result.output) == 1
        assert result.output[0].type == "message"

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
