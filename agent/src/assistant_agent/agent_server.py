import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mlflow.types.agent import ChatAgentChunk, ChatAgentRequest, ChatAgentResponse

from assistant_agent import agent_bot
from assistant_agent.utils import mlflow
from assistant_agent.utils.serving import ChatCompletion

# fastapi run / mlflow が app を import して起動するため、sys.argv は fastapi CLI 等が
# 占有しておりカスタムオプションを混在させられない。そのため環境変数経由でエージェントを指定する。
AGENT_MODULE = os.getenv("AA_AGENT_MODULE", "sample")
OVERWRITE_AGENT_ID = os.getenv("AA_OVERWRITE_AGENT_ID")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncGenerator[None]:
    async with agent_bot.init_harness(AGENT_MODULE, OVERWRITE_AGENT_ID) as (lc_agent, ctx):
        agent_wrapped = mlflow.LangGraphChatAgent(lc_agent, ctx)

        @endpoint.regist(model_id="assistant_agent_v1")
        async def predict(req: ChatAgentRequest) -> ChatAgentResponse:
            return await agent_wrapped.predict_async(req.messages)

        @endpoint.regist_stream(model_id="assistant_agent_v1")
        async def predict_stream(req: ChatAgentRequest) -> AsyncIterator[ChatAgentChunk]:
            async for item in agent_wrapped.predict_stream_async(req.messages):
                yield item

        yield


app = FastAPI(title="Agent Assistant", lifespan=_lifespan)
endpoint = ChatCompletion.bind(app)
