import mlflow
from fastapi import FastAPI

from agent_assistant.agent import agent_wrapped
from agent_assistant.utils.serving import ChatCompletion

mlflow.set_experiment("agent-rag")
mlflow.autolog()

app = FastAPI(title="Agent Assistant")
endpoint = ChatCompletion.bind(app)


@endpoint.regist(model_id="agent_assistant_v1")
def _predict(req):
    """非ストリーミング推論."""
    return agent_wrapped.predict(req.messages)


@endpoint.regist_stream(model_id="agent_assistant_v1")
def _predict_stream(req):
    """ストリーミング推論."""
    return agent_wrapped.predict_stream(req.messages)
