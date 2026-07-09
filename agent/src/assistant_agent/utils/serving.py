import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Awaitable, Callable, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentRequest, ChatAgentResponse
from mlflow.types.chat import ChatMessage, TextContentPart
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request / Response Models (OpenAI Chat Completions API 互換)
# ---------------------------------------------------------------------------
class ChatCompletionRequest(BaseModel):
    """POST /api/chat/completions リクエストボディ."""

    model: str
    messages: list[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    """ChatCompletion の選択肢."""

    index: int
    message: ChatAgentMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    """POST /api/chat/completions レスポンスボディ."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: None = None


class ModelInfo(BaseModel):
    """モデル情報."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """GET /api/models レスポンスボディ."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ChatCompletionChunkChoice(BaseModel):
    """SSE チャンクの選択肢."""

    index: int
    delta: ChatAgentMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """SSE チャンクレスポンス."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---------------------------------------------------------------------------
# ブリッジ関数: ChatMessage ↔ ChatAgentMessage
# ---------------------------------------------------------------------------
def to_chat_agent_messages(messages: list[ChatMessage]) -> list[ChatAgentMessage]:
    """OpenAI ChatMessage を MLflow ChatAgentMessage へ変換."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str) or msg.content is None:
            text = msg.content or ""
        else:
            text = " ".join(part.text for part in msg.content if isinstance(part, TextContentPart))
        result.append(ChatAgentMessage(role=msg.role, content=text))
    return result


def from_chat_agent_response(response: ChatAgentResponse, model: str) -> ChatCompletionResponse:
    """MLflow ChatAgentResponse を OpenAI ChatCompletionResponse へ変換."""
    content = " ".join(m.content for m in response.messages if m.role == "assistant" and m.content)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatAgentMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def from_chat_agent_chunk(chunk_id: str, chunk: ChatAgentChunk, model: str) -> str:
    """MLflow ChatAgentChunk を SSE 行文字列へ変換."""
    data = ChatCompletionChunk(
        id=chunk_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=chunk.delta,
                finish_reason=chunk.finish_reason,
            )
        ],
    )
    return f"data: {json.dumps(data.model_dump(), ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# ChatCompletion ディスパッチャ
# ---------------------------------------------------------------------------
class ChatCompletion:
    """OpenAI Chat Completions API 互換エンドポイントのディスパッチャ."""

    def __init__(self) -> None:
        """Construct ChatCompletion."""
        self._registered_funcs: dict[
            str, Callable[[ChatAgentRequest], Awaitable[ChatAgentResponse]]
        ] = {}
        self._stream_funcs: dict[
            str, Callable[[ChatAgentRequest], AsyncIterator[ChatAgentChunk]]
        ] = {}

    def regist(self, model_id: str) -> Callable:
        """メソッドを Chat Completion API 呼び出し対象へ登録."""

        def _decorator(
            func: Callable[[ChatAgentRequest], Awaitable[ChatAgentResponse]],
        ) -> Callable:
            self._registered_funcs[model_id] = func
            return func

        return _decorator

    def regist_stream(self, model_id: str) -> Callable:
        """メソッドをストリーミング Chat Completion API 呼び出し対象へ登録."""

        def _decorator(
            func: Callable[[ChatAgentRequest], AsyncIterator[ChatAgentChunk]],
        ) -> Callable:
            self._stream_funcs[model_id] = func
            return func

        return _decorator

    async def _invoke_handler(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        messages = to_chat_agent_messages(request.messages)
        agent_request = ChatAgentRequest(messages=messages)

        # クライアントとエージェントが共にストリーミング対応している場合はストリーミングを使用
        if request.model in self._stream_funcs and request.stream:
            stream_handler = self._stream_funcs.get(request.model)
            if stream_handler is None:
                # ストリームハンドラ未登録時は同期フォールバック
                return from_chat_agent_response(
                    await self._registered_funcs[request.model](agent_request), request.model
                )

            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            model_id = request.model

            async def generate() -> AsyncIterator[str]:
                async for chunk in stream_handler(agent_request):
                    yield from_chat_agent_chunk(chunk_id, chunk, model_id)
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        # 指定された model_id のエージェントが無い場合は status_code 503 を返す
        if request.model not in self._registered_funcs:
            raise HTTPException(status_code=503, detail=f"model not registered: {request.model}")
        return from_chat_agent_response(
            await self._registered_funcs[request.model](agent_request), request.model
        )

    def _list_models(self) -> ModelList:
        ids = sorted(set(self._registered_funcs) | set(self._stream_funcs))
        return ModelList(
            data=[ModelInfo(id=mid, created=int(time.time()), owned_by="local") for mid in ids]
        )

    @staticmethod
    def bind(app: FastAPI) -> "ChatCompletion":
        """FastAPI と紐付けた ChatCompletion インスタンスを作成."""
        obj = ChatCompletion()
        app.post("/api/chat/completions", response_model=None)(obj._invoke_handler)
        app.get("/api/models")(obj._list_models)
        return obj
