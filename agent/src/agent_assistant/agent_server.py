from fastapi import FastAPI

from agent_assistant import agents
from agent_assistant.utils.serving import ChatCompletion

app = FastAPI(title="Agent Assistant")
endpoint = ChatCompletion.bind(app)
agent_wrapped = agents.build_agent()


@endpoint.regist(model_id="agent_assistant_v1")
def _predict(req):
    """非ストリーミング推論."""
    return agent_wrapped.predict(req.messages)


@endpoint.regist_stream(model_id="agent_assistant_v1")
def _predict_stream(req):
    """ストリーミング推論."""
    return agent_wrapped.predict_stream(req.messages)
