from collections.abc import Generator
from typing import Any, Optional
from uuid import uuid4

import mlflow
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    convert_to_openai_messages,
)
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent, ResponsesAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)


class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent: CompiledStateGraph[Any, Any, Any, Any], context: Any):
        """Construct LangGraphResponsesAgent."""
        self._agent = agent
        self._context = context

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """エージェントの推論結果を返す."""
        outputs = [
            event.item  # pyright: ignore[reportAttributeAccessIssue]
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """エージェントの推論結果 (Streaming) を返す."""
        cc_msgs = to_chat_completions_input(
            request.input  # pyright: ignore[reportArgumentType]
        )
        for mode, chunk in self._agent.stream(
            {"messages": cc_msgs},
            {"recursion_limit": 100},
            context=self._context,
            # 暫定対処 (ADR-009):
            # "messages" を含めると StreamMessagesHandler (_StreamingCallbackHandler) が
            # 登録され LLM がストリーミングモードになり、on_llm_new_token が content_blocks
            # (list[dict]) を OTel スパン属性に渡すことで警告が大量発生する。
            # mlflow バグ修正後に ["updates", "messages"] へ戻すこと。
            stream_mode=["updates"],
        ):
            if mode == "updates":
                for chunk_state in chunk.values():  # pyright: ignore[reportAttributeAccessIssue]
                    chunk_msgs = chunk_state.get("messages", [])
                    for chunk_msg in chunk_msgs:
                        if isinstance(chunk_msg, AIMessage):
                            # 暫定対処:
                            # mlflow の output_to_responses_items_stream() が内部で呼び出す
                            # create_text_output_item() がコンテンツとして文字列のみを想定しており、
                            # LLM から来る配列が渡す事が不可能であるため、配列内の文字列を連結して
                            # 一つの文字列にして使う
                            chunk_msg.content = "".join(
                                [
                                    item["text"]
                                    for item in chunk_msg.content_blocks
                                    if item["type"] == "text"
                                ]
                            )
                        yield from self.output_to_responses_items_stream(chunk_msgs)

            elif mode == "messages":
                # テキストチャンクのみ出力。他は update 側と重複するため除去
                chunk_msg, _ = chunk
                if isinstance(chunk_msg, AIMessageChunk):
                    # 暫定対処:
                    # mlflow の output_to_responses_items_stream() が内部で呼び出す
                    # create_text_output_item() がコンテンツとして文字列のみを想定しており、
                    # LLM から来る配列が渡す事が不可能であるため、配列内の文字列を連結して
                    # 一つの文字列にして使う
                    chunk_msg.content = "".join(
                        [
                            item["text"]
                            for item in chunk_msg.content_blocks
                            if item["type"] == "text"
                        ]
                    )
                    content = chunk_msg.content
                    if content is None:
                        continue
                    yield ResponsesAgentStreamEvent(
                        **self.create_text_delta(
                            delta=content,
                            item_id=chunk_msg.id,  # pyright: ignore[reportArgumentType]
                        ),
                    )
            else:
                raise ValueError(f"Unknown mode: {mode}")


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph, context: Any):
        """Construct LangGraphChatAgent."""
        self.agent = agent
        self._context = context

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """エージェントの推論結果を返す."""
        req = {"messages": self._convert_messages_to_dict(messages)}
        res = self.agent.invoke(req, {"recursion_limit": 100}, context=self._context)
        assistant_msgs = []
        for item in res["messages"]:
            if isinstance(item, AIMessage):
                msg_dict = convert_to_openai_messages(item)
                if isinstance(msg_dict.get("content"), list):
                    msg_dict["content"] = (
                        "".join(
                            block["text"]
                            for block in msg_dict["content"]
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                        or None
                    )
                assistant_msgs.append(ChatAgentMessage(id=item.id, **msg_dict))
        return ChatAgentResponse(messages=assistant_msgs[-1:])

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """エージェントの推論結果 (Streaming) を返す."""
        request = {"messages": self._convert_messages_to_dict(messages)}
        for mode, chunk in self.agent.stream(
            request,
            {"recursion_limit": 100},
            # 暫定対処 (ADR-009):
            # "messages" を含めると StreamMessagesHandler が登録され on_llm_new_token に
            # content_blocks (list[dict]) が渡ってOTel警告が発生する。
            # LangGraphResponsesAgent と同様の既知問題。mlflow バグ修正後も継続確認すること。
            # stream_mode=["updates", "messages"],
            stream_mode=["messages"],
            context=self._context,
        ):
            if mode == "updates":
                for node_data in chunk.values():  # pyright: ignore[reportAttributeAccessIssue]
                    if "messages" not in node_data:
                        continue
                    for msg_data in node_data["messages"]:
                        msg_id = msg_data.id or str(uuid4())
                        msg_dict = convert_to_openai_messages(msg_data)
                        yield ChatAgentChunk(delta=ChatAgentMessage(id=msg_id, **msg_dict))
            elif mode == "messages":
                chunk_msg, _ = chunk
                if isinstance(chunk_msg, AIMessageChunk):
                    # tool_use/thinking 等の非テキストブロックをフィルタ。
                    # 不完全な tool_use ブロック (input 欠落) が convert_to_openai_messages を
                    # エラーにする問題と、thinking ブロックが list を返す問題を同時に解消する。
                    if isinstance(chunk_msg.content, list):
                        chunk_msg = chunk_msg.model_copy(
                            update={
                                "content": [
                                    item
                                    for item in chunk_msg.content
                                    if isinstance(item, dict) and item.get("type") == "text"
                                ]
                            }
                        )
                    # content_blocks (list[dict]) を自動で string に変換するため
                    # ADR-007 ワークアラウンド不要。
                    msg_dict = convert_to_openai_messages(chunk_msg)
                    if msg_dict["content"]:
                        chunk_id = chunk_msg.id or str(uuid4())
                        yield ChatAgentChunk(delta=ChatAgentMessage(id=chunk_id, **msg_dict))
            else:
                raise ValueError(f"Unknown mode: {mode}")
