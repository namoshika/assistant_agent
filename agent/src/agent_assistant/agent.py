import mlflow.models
from mlflow.genai import agent_server
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent_assistant import connector, context, graph
from agent_assistant.utils.mlflow import LangGraphWrapper

AGENT_NAME = "agent"
ctx = context.build_session()
agent = graph.build_graph(AGENT_NAME, connector.get_llm(), connector.get_tools())
agent_wrapped = LangGraphWrapper(agent, ctx)


# MLflow integration (AgentServer)
@agent_server.invoke()
def invoked_handler(
    request: ResponsesAgentRequest,
) -> ResponsesAgentResponse:
    """Responses API として公開."""
    return agent_wrapped.predict(request)


# MLflow integration (log model)
mlflow.models.set_model(agent_wrapped)
