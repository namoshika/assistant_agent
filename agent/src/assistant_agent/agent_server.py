import os

import mlflow
from fastapi import FastAPI

from assistant_agent import agents
from assistant_agent.utils.serving import ChatCompletion

# トレース用設定
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
if experiment_id is not None:
    mlflow.set_experiment(experiment_id=experiment_id)
else:
    mlflow.set_experiment(experiment_name="agent-rag")
mlflow.bedrock.autolog()  # pyright: ignore[reportPrivateImportUsage]
mlflow.gemini.autolog()  # pyright: ignore[reportPrivateImportUsage]
mlflow.langchain.autolog(run_tracer_inline=True)  # pyright: ignore[reportPrivateImportUsage]

# エージェント初期化
app = FastAPI(title="Agent Assistant")
endpoint = ChatCompletion.bind(app)
agent_wrapped = agents.build_agent()


@endpoint.regist(model_id="assistant_agent_v1")
async def _predict(req):
    """非ストリーミング推論."""
    return await agent_wrapped.predict_async(req.messages)


@endpoint.regist_stream(model_id="assistant_agent_v1")
async def _predict_stream(req):
    """ストリーミング推論."""
    async for item in agent_wrapped.predict_stream_async(req.messages):
        yield item
