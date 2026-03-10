from collections.abc import Generator
from typing import Any
from langchain_core.messages import AIMessageChunk, AIMessage
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)


class LangGraphWrapper(ResponsesAgent):
    def __init__(self, agent: CompiledStateGraph[Any, Any, Any, Any], context: Any):
        self._agent = agent
        self._context = context

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item  # pyright: ignore[reportAttributeAccessIssue]
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input(
            request.input  # pyright: ignore[reportArgumentType]
        )
        for mode, chunk in self._agent.stream(
            {"messages": cc_msgs},
            {"recursion_limit": 100},
            context=self._context,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for (
                    chunk_state
                ) in chunk.values():  # pyright: ignore[reportAttributeAccessIssue]
                    chunk_msgs = chunk_state.get("messages", [])
                    for chunk_msg in chunk_msgs:
                        if isinstance(chunk_msg, AIMessage):
                            # 暫定対処:
                            # mlflow の output_to_responses_items_stream() が内部で呼び出す
                            # create_text_output_item() がコンテンツとして文字列のみを想定しており、LLM から
                            # 来る配列が渡す事が不可能であるため、配列内の文字列を連結して一つの文字列にして使う
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
                    # create_text_output_item() がコンテンツとして文字列のみを想定しており、LLM から
                    # 来る配列が渡す事が不可能であるため、配列内の文字列を連結して一つの文字列にして使う
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
