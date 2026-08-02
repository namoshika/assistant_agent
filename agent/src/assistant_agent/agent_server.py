import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from langchain_core.messages import BaseMessage
from mlflow.types.agent import ChatAgentMessage, ChatAgentRequest, ChatAgentResponse

from assistant_agent import agent_bot
from assistant_agent.utils.serving import ChatCompletion

logger = logging.getLogger(__name__)

# トレース用設定
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")
if MLFLOW_EXPERIMENT_ID is not None:
    mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
else:
    mlflow.set_experiment(experiment_name="agent-rag")
mlflow.bedrock.autolog()  # pyright: ignore[reportPrivateImportUsage]
mlflow.gemini.autolog()  # pyright: ignore[reportPrivateImportUsage]
mlflow.langchain.autolog(run_tracer_inline=True)  # pyright: ignore[reportPrivateImportUsage]


def _extract_text_content(msg: BaseMessage) -> str | None:
    """msg.content が content_blocks（list[dict]）の場合、text ブロックのみを連結して返す.

    text ブロックが1件もない場合（例: thinking ブロックのみの応答）は None を返す。
    """
    if not isinstance(msg.content, list):
        return msg.content
    return (
        "".join(
            block["text"]
            for block in msg.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        or None
    )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncGenerator[None]:
    async with agent_bot.init_harness() as sync_request_channel:

        @endpoint.regist(model_id="assistant_agent_v1")
        async def predict(req: ChatAgentRequest) -> ChatAgentResponse:
            if not req.messages or not req.messages[-1].content:
                raise HTTPException(status_code=400, detail="messages must be non-empty")
            content = req.messages[-1].content
            msg_out = await sync_request_channel.emit_and_wait(
                content, channel_name="Chat Completions API Channel"
            )
            text_content = _extract_text_content(msg_out)
            if text_content is None:
                logger.error(f"Agent response has no text content: {msg_out!r}")
                raise HTTPException(status_code=500, detail="agent returned no text content")
            return ChatAgentResponse(
                messages=[
                    ChatAgentMessage(
                        id=msg_out.id or str(uuid.uuid7()), role="assistant", content=text_content
                    )
                ]
            )

        yield


app = FastAPI(title="Agent Assistant", lifespan=_lifespan)
endpoint = ChatCompletion.bind(app)
